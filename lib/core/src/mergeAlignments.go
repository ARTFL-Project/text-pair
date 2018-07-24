package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

var stopwords = make(map[string]bool)
var frequencyRegex = regexp.MustCompile(`(\d+)`)
var passageRegex = regexp.MustCompile(`^\d+(.*)`)
var tokenRegex = regexp.MustCompile(`(?i)([\w]+)|([\.?!\-])`)
var punctRegex = regexp.MustCompile(`[\.?!,;\-:\(\)']`)
var numRegex = regexp.MustCompile(`\d`)

var passageGroupMap map[string]int

type passagePosition struct {
	startByte int
	endByte   int
	groupID   int
}

type passageGroup struct {
	filename      string
	startByte     int
	endByte       int
	sourcePassage string
	groupID       int
	matches       int
}

func passageGroupInit(passage map[string]string, groupID *int, mergedTargetPassages map[string][]*passagePosition) *passageGroup {
	startByte, _ := strconv.Atoi(passage["source_start_byte"])
	endByte, _ := strconv.Atoi(passage["source_end_byte"])
	*groupID++
	targetStartByte, _ := strconv.Atoi(passage["target_start_byte"])
	targetEndByte, _ := strconv.Atoi(passage["target_end_byte"])
	currentTarget := &passagePosition{targetStartByte, targetEndByte, *groupID}
	mergedTargetPassages[passage["target_doc_id"]] = append(mergedTargetPassages[passage["target_doc_id"]], currentTarget)
	passageGroupMap[passage["passage_id"]] = *groupID
	return &passageGroup{passage["source_filename"], startByte, endByte, passage["source_passage"], *groupID, 2}
}

func passageGroupUpdate(currentGroup *passageGroup, passage map[string]string, groupID *int, mergedTargetPassages map[string][]*passagePosition) {
	endByte, _ := strconv.Atoi(passage["source_end_byte"])
	if endByte > currentGroup.endByte {
		currentGroup.endByte = endByte
	}
	targetStartByte, _ := strconv.Atoi(passage["target_start_byte"])
	targetEndByte, _ := strconv.Atoi(passage["target_end_byte"])
	currentTarget := &passagePosition{targetStartByte, targetEndByte, *groupID}
	mergedTargetPassages[passage["target_doc_id"]] = append(mergedTargetPassages[passage["target_doc_id"]], currentTarget)
	currentGroup.matches += 2
	passageGroupMap[passage["passage_id"]] = *groupID
}

func getStopwords(fileLocation string) map[string]bool {
	file, err := os.Open(fileLocation)

	if err != nil {
		panic(err)
	}

	defer file.Close()

	reader := bufio.NewReader(file)
	stopwords := make(map[string]bool)
	var line string
	for {
		line, err = reader.ReadString('\n')
		if err != nil {
			break
		}
		stopwords[strings.TrimSpace(line)] = true
	}
	return stopwords
}

func tokenize(passage string) (tokens []string) {
	passage = strings.ToLower(passage)
	rawTokens := tokenRegex.FindAllString(passage, -1)
	for _, value := range rawTokens {
		if utf8.RuneCountInString(value) > 3 && !punctRegex.MatchString(value) && !numRegex.MatchString(value) {
			if _, ok := stopwords[value]; !ok {
				tokens = append(tokens, value)
			}
		}
	}
	return
}

func percentBuilder(count int) map[int]int {
	percent := make(map[int]int)
	percentIncrement := int(math.Round(float64(count) / 100))
	incrementValue := 0
	for i := 1; i < 101; i++ {
		incrementValue += percentIncrement
		percent[incrementValue] = i
	}
	return percent
}

func hash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

func extractFields(fields map[string]string, prefix string) map[string]string {
	extractedFields := make(map[string]string)
	for key, value := range fields {
		if strings.HasPrefix(key, prefix) {
			extractedFields[key] = value
		}
	}
	return extractedFields
}

func mergeSourcePassages(passages []map[string]string, mergedSourcePassages []*passageGroup, groupID *int, mergedTargetPassages map[string][]*passagePosition) {
	sort.Slice(passages, func(i, j int) bool {
		if passages[i]["start_byte"] < passages[j]["start_byte"] {
			return true
		} else if passages[i]["start_byte"] == passages[j]["start_byte"] {
			if passages[i]["end_byte"] > passages[j]["end_byte"] {
				return true
			}
		}
		return false
	})
	currentGroup := &passageGroup{}
	for _, passage := range passages {
		if currentGroup.matches == 0 {
			currentGroup = passageGroupInit(passage, groupID, mergedTargetPassages)
			continue
		}
		currentStartByte, _ := strconv.Atoi(passage["source_start_byte"])
		if currentStartByte < currentGroup.endByte {
			passageGroupUpdate(currentGroup, passage, groupID, mergedTargetPassages)
		} else {
			filename := passage["source_filename"]
			currentGroup.sourcePassage = getText(&filename, int32(currentGroup.startByte), int32(currentGroup.endByte))
			currentGroup = passageGroupInit(passage, groupID, mergedTargetPassages)
		}
	}
	if currentGroup.matches != 0 {
		mergedSourcePassages = append(mergedSourcePassages, currentGroup)
	}
}

func mergeAlignments(outputPath string, numThreads int, alignmentCount int) {
	fmt.Printf("\nMerging all %d alignments\n", alignmentCount)

	file, err := os.Open(outputPath + "/" + "alignment.results")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)

	passageGroupMap = make(map[string]int)

	// First pass over results to merge overlapping source passages
	count := 0
	groupID := 0
	percent := percentBuilder(alignmentCount)
	docID := -1
	var line string
	var passages []map[string]string
	var mergedSourcePassages []*passageGroup
	mergedTargetPassages := make(map[string][]*passagePosition) // key is source docID
	for {
		line, err = reader.ReadString('\n')
		if err != nil {
			break
		}
		count++
		line = strings.TrimSpace(line)
		fields := make(map[string]string)
		json.Unmarshal([]byte(line), &fields)

		// Check if a section of current source passage wasn't merged as a target passage in a previous
		// passage group
		sourceDocID := fields["source_doc_id"]
		match := false
		if _, ok := mergedTargetPassages[sourceDocID]; ok {
			startByte, _ := strconv.Atoi(fields["source_start_byte"])
			endByte, _ := strconv.Atoi(fields["source_end_byte"])
			for _, localTarget := range mergedTargetPassages[sourceDocID] {
				if (localTarget.startByte >= startByte && localTarget.startByte < endByte) ||
					(localTarget.startByte < startByte && localTarget.endByte > startByte) {
					sourcePassageMatch := &passagePosition{startByte, endByte, localTarget.groupID}
					targetStartByte, _ := strconv.Atoi(fields["target_start_byte"])
					targetEndByte, _ := strconv.Atoi(fields["target_end_byte"])
					targetPassageMatch := &passagePosition{targetStartByte, targetEndByte, localTarget.groupID}
					mergedTargetPassages[sourceDocID] = append(mergedTargetPassages[sourceDocID], sourcePassageMatch, targetPassageMatch)
					passageGroupMap[fields["passage_id"]] = localTarget.groupID
					match = true
					break
				}
			}
		}
		if match { // We do not store this passage since it was already matched previously
			if _, ok := percent[count]; ok {
				fmt.Printf("\rGrouping passages... %d %%", percent[count])
			}
			continue
		}

		currentDocID, _ := strconv.Atoi(sourceDocID)
		if docID != currentDocID && docID != -1 {
			mergeSourcePassages(passages, mergedSourcePassages, &groupID, mergedTargetPassages)
			passages = []map[string]string{}
		}

		docID = currentDocID
		passages = append(passages, fields)

		if _, ok := percent[count]; ok {
			fmt.Printf("\rGrouping passages... %d %%", percent[count])
		}
	}
	if len(passages) > 0 {
		mergeSourcePassages(passages, mergedSourcePassages, &groupID, mergedTargetPassages)
		passages = []map[string]string{}
	}

	fmt.Printf("\rGrouping passages... %d groups found.\n", groupID)

	// Second pass over results to store groupIDs
	file, err = os.Open(outputPath + "/" + "alignment.results")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader = bufio.NewReader(file)

	outputFile, _ := os.Create(outputPath + "/" + "merge_alignment_results.txt")

	fmt.Print("Saving results...")
	for {
		line, err = reader.ReadString('\n')
		if err != nil {
			break
		}
		fields := make(map[string]string)
		json.Unmarshal([]byte(line), &fields)
		fields["group_id"] = strconv.Itoa(passageGroupMap[fields["passage_id"]])
		jsonString, _ := json.Marshal(fields)
		jsonString = append(jsonString, "\n"...)
		outputFile.Write(jsonString)
	}
	outputFile.Sync()
	outputFile.Close()
	os.Remove(outputPath + "/" + "alignment.results")
	os.Rename(outputPath+"/"+"merge_alignment_results.txt", outputPath+"/"+"alignment.results")
	fmt.Println(" done.")
}
