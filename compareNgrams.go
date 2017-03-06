package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"regexp"
)

type docIndex struct {
	DocID  string
	Ngrams map[int32][]indexedNgram
}

type indexedNgram []int32

type ngramMatch struct {
	source indexedNgram
	target indexedNgram
	ngram  int32
}

type matchingParams struct {
	matchingWindowSize            int32
	maxGap                        int32
	minimumMatchingNgrams         int32
	minimumMatchingNgramsInWindow int32
	commonNgramsLimit             float32
	percentMatching               int32
	minimumMatchingNgramsInDocs   int32
	contextSize                   int32
	banalNgrams                   int32
}

type MatchValues struct {
	inAlignment               bool
	matchesInCurrentAlignment int32
	matchesInCurrentWindow    int32
	sourceAnchor              int32
	lastSourcePosition        int32
	targetAnchor              int32
	lastTargetPosition        int32
	previousSourceIndex       int32
	commonNgramMatches        int32
	maxSourceGap              int32
	maxTargetGap              int32
	sourceWindowBoundary      int32
	targetWindowBoundary      int32
	currentAlignment          Alignment
	firstMatch                []indexedNgram
	lastMatch                 []indexedNgram
}

// Alignment is the matching representation
type Alignment struct {
	source position
	target position
}

type position struct {
	startByte  int32
	endByte    int32
	startIndex int32
	endIndex   int32
}

type alignmentsPerDoc struct {
	docID   string
	matches []Alignment
}

// CombinedAlignments holds all alignments for a single source doc
type CombinedAlignments struct {
	sourceID   string
	alignments []alignmentsPerDoc
}

// Pair is a data structure to hold a key/value pair.
type Pair struct {
	Key   int32
	Value int
}

// PairList is a slice of Pairs that implements sort. Interface to sort by Value.
type PairList []Pair

func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value > p[j].Value }

// A function to turn a map into a PairList, then sort and return it.
func sortMapByValue(m map[int32]int) PairList {
	p := make(PairList, len(m))
	i := 0
	for k, v := range m {
		p[i] = Pair{k, v}
	}
	sort.Sort(p)
	return p
}

// PairList is a slice of Pairs that implements sort. Interface to sort by Value.
type matchingNgrams []ngramMatch

func (p matchingNgrams) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p matchingNgrams) Len() int      { return len(p) }
func (p matchingNgrams) Less(i, j int) bool {
	if p[i].source[0] < p[j].source[0] {
		return true
	} else if p[i].source[0] > p[j].source[0] {
		return false
	}
	return p[i].target[0] < p[j].target[0]
}

var matchingDefaults = matchingParams{20, 10, 4, 4, 0.75, 10, 4, 300, 25}

var outputFormat = "tab"

var tags = regexp.MustCompile("<[^>]*?>")
var brokenBeginTags = regexp.MustCompile("^[^<]*?>")
var brokenEndTags = regexp.MustCompile("<[^>]*?$")
var spaces = regexp.MustCompile(" +")

func checkErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func getFiles(filePath string) []docIndex {
	if filePath == "" {
		return []docIndex{}
	}
	if !strings.HasPrefix(filePath, "/") {
		filePath = "./" + filePath
	}
	directory, err := os.Open(filePath)
	checkErr(err)
	files, err := directory.Readdir(-1)
	checkErr(err)
	jsonFiles := make([]docIndex, 0)
	for _, fileInfo := range files {
		if !fileInfo.IsDir() {
			file := filepath.Join(filePath, fileInfo.Name())
			jsonFile := openJSONDoc(&file)
			docID := strings.Replace(fileInfo.Name(), ".json", "", 1)
			doc := docIndex{docID, jsonFile}
			jsonFiles = append(jsonFiles, doc)
		}
	}
	return jsonFiles
}

func parseFlags() ([]docIndex, []docIndex, map[string]map[string]string, map[string]map[string]string, int, *string) {
	sourceFilesArg := flag.String("source_files", "", "source files location")
	targetFilesArg := flag.String("target_files", "", "target files location")
	threadsArg := flag.Int("threads", 4, "number of threads to use")
	sourceMetadataArg := flag.String("source_metadata", "", "path to source metadata")
	targetMetadataArg := flag.String("target_metadata", "", "path to target metadata")
	outputPath := flag.String("output_path", "./", "output path for results")
	flag.Parse()
	numThreads := *threadsArg
	fmt.Println("Reading JSON files...")
	sourceFiles := getFiles(*sourceFilesArg)
	targetFiles := getFiles(*targetFilesArg)
	fmt.Println("Loading metadata...")
	sourceMetadata := openJSONMetadata(sourceMetadataArg)
	targetMetadata := openJSONMetadata(targetMetadataArg)
	return sourceFiles, targetFiles, sourceMetadata, targetMetadata, numThreads, outputPath
}

func openJSONMetadata(fileLocation *string) map[string]map[string]string {
	if *fileLocation == "" {
		return map[string]map[string]string{}
	}
	var filePath string
	if !strings.HasPrefix(*fileLocation, "/") {
		filePath = "./" + *fileLocation
	} else {
		filePath = *fileLocation
	}
	jsonFile, err := ioutil.ReadFile(filePath)
	checkErr(err)
	metadata := make(map[string]map[string]string)
	json.Unmarshal(jsonFile, &metadata)
	return metadata
}

func openJSONDoc(fileLocation *string) map[int32][]indexedNgram {
	jsonFile, err := ioutil.ReadFile(*fileLocation)
	checkErr(err)
	doc := make(map[int32][]indexedNgram)
	json.Unmarshal(jsonFile, &doc)
	return doc
}

func getIntersection(sourceFile *docIndex, targetFile *docIndex) map[int32]int {
	intersectCount := make(map[int32]int)
	if len(sourceFile.Ngrams) < len(targetFile.Ngrams) {
		for ngram := range sourceFile.Ngrams {
			if _, ok := targetFile.Ngrams[ngram]; ok {
				intersectCount[ngram] = len(sourceFile.Ngrams[ngram]) + len(targetFile.Ngrams[ngram])
			}
		}
	} else {
		for ngram := range targetFile.Ngrams {
			if _, ok := sourceFile.Ngrams[ngram]; ok {
				intersectCount[ngram] = len(sourceFile.Ngrams[ngram]) + len(targetFile.Ngrams[ngram])
			}
		}
	}
	return intersectCount
}

func getMostCommonNgrams(intersectionCount map[int32]int) map[int32]bool {
	sortedIntersection := sortMapByValue(intersectionCount)
	mostCommonNgrams := make(map[int32]bool)
	var count int32
	for _, pair := range sortedIntersection {
		if pair.Value == 2 {
			break
		}
		mostCommonNgrams[pair.Key] = true
		count++
		if count == matchingDefaults.banalNgrams {
			break
		}
	}
	return mostCommonNgrams
}

func writeAligments(combinedAlignments *CombinedAlignments, sourceDocID *string, sourceMetadata map[string]map[string]string,
	targetMetadata map[string]map[string]string, f *os.File) {
	combinedOutput := make([]string, len(combinedAlignments.alignments))
	_, sourceValues := mapToSliceOfSlices(sourceMetadata[*sourceDocID])
	for _, alignments := range combinedAlignments.alignments {
		_, targetValues := mapToSliceOfSlices(targetMetadata[alignments.docID])
		matchOutputs := []string{}
		for _, alignment := range alignments.matches {
			fields := []string{}
			fields = append(fields, sourceValues...)
			fields = append(fields, []string{strconv.Itoa(int(alignment.source.startByte)), strconv.Itoa(int(alignment.source.endByte))}...)
			fields = append(fields, alignmentToText(&alignment.source, sourceMetadata[*sourceDocID]["filename"])...)
			fields = append(fields, targetValues...)
			fields = append(fields, []string{strconv.Itoa(int(alignment.target.startByte)), strconv.Itoa(int(alignment.target.endByte))}...)
			fields = append(fields, alignmentToText(&alignment.target, targetMetadata[alignments.docID]["filename"])...)
			matchOutputs = append(matchOutputs, strings.Join(fields, "\t"))
			combinedOutput = append(combinedOutput, strings.Join(fields, "\t"))
		}
	}
	f.WriteString(strings.Join(combinedOutput, "\n"))
}

func mapToSliceOfSlices(metadata map[string]string) ([]string, []string) {
	keys := make([]string, len(metadata))
	values := make([]string, len(metadata))
	i := 0
	for k := range metadata {
		values[i] = metadata[k]
		keys[i] = k
		i++
	}
	return keys, values
}

func alignmentToText(alignment *position, filename string) []string {
	beforeContext := getText(&filename, alignment.startByte-int32(matchingDefaults.contextSize), alignment.startByte)
	matchingPassage := getText(&filename, alignment.startByte, alignment.endByte)
	afterContext := getText(&filename, alignment.endByte, alignment.endByte+int32(matchingDefaults.contextSize))
	passages := []string{beforeContext, matchingPassage, afterContext}
	return passages
}

func getText(fileLocation *string, startByte int32, endByte int32) string {
	f, err := os.Open(*fileLocation)
	checkErr(err)
	_, err = f.Seek(int64(startByte), 0)
	checkErr(err)
	passage := make([]byte, endByte-startByte)
	_, err = f.Read(passage)
	checkErr(err)
	text := string(passage)
	text = tags.ReplaceAllString(text, "")
	text = brokenBeginTags.ReplaceAllString(text, "")
	text = brokenEndTags.ReplaceAllString(text, "")
	text = strings.Replace(text, "\t", " ", -1)
	text = strings.Replace(text, "\n", " ", -1)
	text = spaces.ReplaceAllString(text, " ")
	f.Close()
	return text
}

func createOutputFile(outputPath *string, sourceMetadata map[string]map[string]string, targetMetadata map[string]map[string]string) *os.File {
	os.MkdirAll(*outputPath, 0755)
	mergedOutput, err := os.Create(fmt.Sprintf(filepath.Join(*outputPath, "alignments_results.tab")))
	checkErr(err)
	var firstSourceKey string
	for sourceKey := range sourceMetadata {
		firstSourceKey = sourceKey
		break
	}
	sourceFields, _ := mapToSliceOfSlices(sourceMetadata[firstSourceKey])
	firstRow := sourceFields
	firstRow = append(firstRow, []string{"source_start_byte", "source_end_byte"}...)
	firstRow = append(firstRow, []string{"source_context_before", "source_passage", "source_context_after"}...)
	var firstTargetKey string
	for targetKey := range targetMetadata {
		firstTargetKey = targetKey
		break
	}
	targetNames, _ := mapToSliceOfSlices(targetMetadata[firstTargetKey])
	firstRow = append(firstRow, targetNames...)
	firstRow = append(firstRow, []string{"target_start_byte", "target_end_byte"}...)
	firstRow = append(firstRow, []string{"target_context_before", "target_passage", "target_context_after"}...)
	mergedOutput.WriteString(strings.Join(firstRow, "\t") + "\n")
	return mergedOutput
}

func main() {
	sourceFiles, targetFiles, sourceMetadata, targetMetadata, numThreads, outputPath := parseFlags()
	sourceAgainstSource := false
	sourceFileDone := make(map[string]bool)
	if len(targetFiles) == 0 {
		targetFiles = sourceFiles
		targetMetadata = sourceMetadata
		sourceAgainstSource = true
	}
	mergedOutput := createOutputFile(outputPath, sourceMetadata, targetMetadata)

	counts := 0
	targetLength := len(targetFiles)
	var wait sync.WaitGroup
	for _, sourceFile := range sourceFiles {
		fmt.Printf("Comparing source file %s to all...\n", sourceFile.DocID)
		combinedAlignments := CombinedAlignments{sourceFile.DocID, []alignmentsPerDoc{}}
		c := make(chan []alignmentsPerDoc, numThreads)
		wait.Add(numThreads)
		start := 0
		increment := targetLength/numThreads + 1
		for i := 0; i < numThreads; i++ {
			end := increment * (i + 1)
			if end > targetLength {
				end = targetLength - 1
			}
			splitTargets := targetFiles[start:end]
			start += increment
			if start > targetLength-1 {
				start = targetLength - 1
			}
			go func(splitTargets []docIndex, sourceAgainstSource bool, sourceMetadata map[string]map[string]string, targetMetadata map[string]map[string]string, sourceFileDone map[string]bool) {
				defer wait.Done()
				localAlignments := []alignmentsPerDoc{}
				for _, targetFile := range splitTargets {
					if sourceAgainstSource {
						if sourceMetadata[sourceFile.DocID]["filename"] == targetMetadata[targetFile.DocID]["filename"] {
							continue
						} else if _, ok := sourceFileDone[targetFile.DocID]; ok {
							continue
						}
					}
					sourceTargetIntersection := getIntersection(&sourceFile, &targetFile)
					if int32(len(sourceTargetIntersection)) < matchingDefaults.minimumMatchingNgramsInDocs {
						continue
					}
					mostCommonNgrams := getMostCommonNgrams(sourceTargetIntersection)
					var matches = matchingNgrams{}
					for ngram := range sourceTargetIntersection {
						for _, sourceNgramIndex := range sourceFile.Ngrams[ngram] {
							for _, targetNgramIndex := range targetFile.Ngrams[ngram] {
								matches = append(matches, ngramMatch{sourceNgramIndex, targetNgramIndex, ngram})
							}
						}
					}
					sort.Sort(matchingNgrams(matches))
					alignments := make([]Alignment, 0)
					matchValues := MatchValues{}
					matchValues.lastSourcePosition = 0
					matchValues.inAlignment = false
					for matchIndex, currentAnchor := range matches {
						if currentAnchor.source[0] < matchValues.lastSourcePosition {
							continue
						}
						matchValues.sourceAnchor = currentAnchor.source[0]
						matchValues.sourceWindowBoundary = matchValues.sourceAnchor + matchingDefaults.matchingWindowSize
						matchValues.lastSourcePosition = matchValues.sourceAnchor
						matchValues.maxSourceGap = matchValues.lastSourcePosition + matchingDefaults.maxGap
						matchValues.targetAnchor = currentAnchor.target[0]
						matchValues.targetWindowBoundary = matchValues.targetAnchor + matchingDefaults.matchingWindowSize
						matchValues.lastTargetPosition = matchValues.targetAnchor
						matchValues.maxTargetGap = matchValues.lastTargetPosition + matchingDefaults.maxGap
						matchValues.inAlignment = true
						matchValues.previousSourceIndex = matchValues.sourceAnchor
						matchValues.firstMatch = []indexedNgram{currentAnchor.source, currentAnchor.target}
						matchValues.matchesInCurrentAlignment = 1
						matchValues.matchesInCurrentWindow = 1
						matchValues.commonNgramMatches = 0
						if _, ok := mostCommonNgrams[currentAnchor.ngram]; ok {
							matchValues.commonNgramMatches++
						}
						matchValues.lastMatch = []indexedNgram{currentAnchor.source, currentAnchor.target}
					innerMatchingLoop:
						for _, match := range matches[matchIndex+1:] {
							source, target := match.source, match.target
							// we skip source_match if the same as before and we only want targets that are after last target match
							if source[0] == matchValues.previousSourceIndex || target[0] <= matchValues.lastTargetPosition {
								continue
							}
							if source[0] > matchValues.maxSourceGap || target[0] > matchValues.maxTargetGap {
								// println("Failed", source[0], matchValues.lastSourcePosition+matchingDefaults.maxGap, target[0], matchValues.lastTargetPosition+matchingDefaults.maxGap)
								// println(firstStep, secondStep)
								matchValues.inAlignment = false
							}
							if source[0] > matchValues.sourceWindowBoundary || target[0] > matchValues.targetWindowBoundary {
								if matchValues.matchesInCurrentWindow < matchingDefaults.minimumMatchingNgramsInWindow {
									matchValues.inAlignment = false
								} else {
									if source[0] > matchValues.maxSourceGap || target[0] > matchValues.maxTargetGap {
										matchValues.inAlignment = false
									} else {
										matchValues.sourceAnchor = source[0]
										matchValues.sourceWindowBoundary = matchValues.sourceAnchor + matchingDefaults.matchingWindowSize
										matchValues.targetAnchor = target[0]
										matchValues.targetWindowBoundary = matchValues.targetAnchor + matchingDefaults.matchingWindowSize
										matchValues.matchesInCurrentWindow = 0
									}
								}
							}
							if !matchValues.inAlignment {
								if float32(matchValues.commonNgramMatches/matchValues.matchesInCurrentAlignment) < matchingDefaults.commonNgramsLimit {
									if matchValues.matchesInCurrentAlignment >= matchingDefaults.minimumMatchingNgramsInWindow {
										matchValues.currentAlignment.source = position{matchValues.firstMatch[0][1], matchValues.lastMatch[0][2], matchValues.firstMatch[0][0], matchValues.lastMatch[0][0]}
										matchValues.currentAlignment.target = position{matchValues.firstMatch[1][1], matchValues.lastMatch[1][2], matchValues.firstMatch[1][0], matchValues.lastMatch[1][0]}
										alignments = append(alignments, matchValues.currentAlignment)
										// fmt.Println(matchValues.currentAlignment.source.startIndex, matchValues.currentAlignment.source.endIndex)
										// Looking for small match within max_gap
									} else if (matchValues.lastMatch[0][0]-currentAnchor.source[0]) <= matchingDefaults.maxGap && matchValues.matchesInCurrentAlignment >= matchingDefaults.minimumMatchingNgrams {
										matchValues.currentAlignment.source = position{matchValues.firstMatch[0][1], matchValues.lastMatch[0][2], matchValues.firstMatch[0][0], matchValues.lastMatch[0][0]}
										matchValues.currentAlignment.target = position{matchValues.firstMatch[1][1], matchValues.lastMatch[1][2], matchValues.firstMatch[1][0], matchValues.lastMatch[1][0]}
										alignments = append(alignments, matchValues.currentAlignment)
										// fmt.Println(matchValues.currentAlignment.source.startIndex, matchValues.currentAlignment.source.endIndex)
									}
								}
								matchValues.lastSourcePosition = matchValues.lastMatch[0][0] + 1 // Make sure we start the next match at index that follows last source match
								// if matchValues.currentAlignment.source.endByte != 0 {
								// 	fmt.Println("Failed", source[0], target[0], matchValues.currentAlignment.source.endIndex, matchValues.currentAlignment.target.endIndex)
								// }
								break innerMatchingLoop
							}
							matchValues.lastSourcePosition = source[0]
							matchValues.maxSourceGap = matchValues.lastSourcePosition + matchingDefaults.maxGap
							matchValues.lastTargetPosition = target[0]
							matchValues.maxTargetGap = matchValues.lastTargetPosition + matchingDefaults.maxGap
							matchValues.previousSourceIndex = source[0]
							matchValues.matchesInCurrentWindow++
							matchValues.matchesInCurrentAlignment++
							matchValues.lastMatch = []indexedNgram{source, target} // save last matching ngrams
							if _, ok := mostCommonNgrams[match.ngram]; ok {
								matchValues.commonNgramMatches++
							}
						}
					}
					// Add current alignment if not already done
					if matchValues.inAlignment && matchValues.matchesInCurrentAlignment >= matchingDefaults.minimumMatchingNgrams {
						matchValues.currentAlignment.source = position{matchValues.firstMatch[0][1], matchValues.lastMatch[0][2], matchValues.firstMatch[0][0], matchValues.lastMatch[0][0]}
						matchValues.currentAlignment.target = position{matchValues.firstMatch[1][1], matchValues.lastMatch[1][2], matchValues.firstMatch[1][0], matchValues.lastMatch[1][0]}
						alignments = append(alignments, matchValues.currentAlignment)
					}
					counts += len(alignments)
					if len(alignments) > 0 {
						localAlignments = append(localAlignments, alignmentsPerDoc{targetFile.DocID, alignments})
					}
				}
				c <- localAlignments
			}(splitTargets, sourceAgainstSource, sourceMetadata, targetMetadata, sourceFileDone)
		}
		wait.Wait()
		for i := 0; i < numThreads; i++ {
			localCombinedAlignments := <-c
			if len(localCombinedAlignments) > 0 {
				combinedAlignments.alignments = append(combinedAlignments.alignments, localCombinedAlignments...)
			}
		}
		if len(combinedAlignments.alignments) > 0 {
			writeAligments(&combinedAlignments, &sourceFile.DocID, sourceMetadata, targetMetadata, mergedOutput)
		}
		if sourceAgainstSource {
			sourceFileDone[sourceFile.DocID] = true
		}
	}
	mergedOutput.Sync()
	mergedOutput.Close()
	fmt.Printf("%d results...\n", counts)
}
