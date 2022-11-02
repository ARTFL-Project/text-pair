package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type sortedFile struct {
	docID  string
	sortID int
}

type docIndex struct {
	DocID       string
	Ngrams      map[int32][]indexedNgram
	NgramLength int
	SortID      int
}

type indexedNgram struct {
	index     int32
	startByte int32
	endByte   int32
}

type ngramMatch struct {
	source indexedNgram
	target indexedNgram
	ngram  int32
}

type matchingParams struct {
	matchingWindowSize            int32
	maxGap                        int32
	flexGap                       bool
	minimumMatchingNgrams         int32
	minimumMatchingNgramsInWindow int32
	commonNgramsLimit             float32
	minimumMatchingNgramsInDocs   int
	contextSize                   int32
	banalNgrams                   int
	mergeOnByteDistance           bool
	mergeOnNgramDistance          bool
	passageDistanceMultiplier     float64
	duplicateThreshold            float64
	sourceBatch                   int
	targetBatch                   int
	outputPath                    string
	numThreads                    int
	sortingField                  string
	debug                         bool
}

type matchValues struct {
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
	previousAlignment         Alignment
	firstMatch                []indexedNgram
	lastMatch                 []indexedNgram
	debug                     []string // the string is the original ngram
}

// Alignment is the matching representation
type Alignment struct {
	source              position
	target              position
	totalMatchingNgrams int32
	banality            bool
}

type position struct {
	startByte       int32
	endByte         int32
	startNgramIndex int32
	endNgramIndex   int32
}

type alignmentsPerDoc struct {
	docID      string
	matches    []Alignment
	duplicates string
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

var tags = regexp.MustCompile("<[^>]*?>")
var brokenBeginTags = regexp.MustCompile("^[^<]*?>")
var brokenEndTags = regexp.MustCompile("<[^>]*?$")
var spaces = regexp.MustCompile(" +")
var spaceChars = regexp.MustCompile(`[\s\r\n\t]+`)
var tabEntities = regexp.MustCompile("(&#9;)+")
var cleanStart = regexp.MustCompile(`^\S+ `)
var cleanEnd = regexp.MustCompile(` \S+$`)

func main() {
	sourceFiles, targetFiles, sourceMetadata, targetMetadata, commonNgrams, config, ngramIndex := parseFlags()
	_ = alignPassages(sourceFiles, targetFiles, sourceMetadata, targetMetadata, commonNgrams, config, ngramIndex)
	// mergeAlignments(config, counts)
}

func parseFlags() ([]sortedFile, []sortedFile, map[string]map[string]string, map[string]map[string]string, map[int32]bool, *matchingParams, map[int32]string) {
	outputPath := flag.String("output_path", "./output", "output path for results")
	ngramIndexLocation := flag.String("ngram_index", "", "location of ngram index used for debugging. Should be the source or target index, not matter which since it'll be used for common ngrams")
	sourceFilesArg := flag.String("source_files", "", "source files location")
	targetFilesArg := flag.String("target_files", "", "target files location")
	threadsArg := flag.Int("threads", 4, "number of threads to use")
	sourceMetadataArg := flag.String("source_metadata", "", "path to source metadata")
	targetMetadataArg := flag.String("target_metadata", "", "path to target metadata")
	sortField := flag.String("sort_by", "year", "metadata field used to sort files in ascending order")
	sourceBatch := flag.Int("source_batch", 1, "Split the source files into n number of batches: useful when RAM usage is a concern")
	targetBatch := flag.Int("target_batch", 1, "Split the target files into n number of batches: useful when RAM usage is a concern")
	sourceCommonNgramsArg := flag.String("source_common_ngrams", "", "path to a text file containing the most common ngrams in source files")
	targetCommonNgramsArg := flag.String("target_common_ngrams", "", "path to a text file containing the most common ngrams in target files")
	mostCommonNgramThreshold := flag.Int("most_common_ngram_threshold", 1000, "take the n most common ngrams from source and target common ngrams")
	commonNgramsLimit := flag.Int("common_ngrams_limit", 25, "percentage of common ngrams to dismiss a match as banal")
	matchingWindowSize := flag.Int("matching_window_size", 30, "size of sliding window for matches")
	maxGap := flag.Int("max_gap", 15, "maximum gap between two matching ngrams")
	flexGap := flag.Bool("flex_gap", false, "Gradually increment the max_gap once minimum_matching_ngrams is met")
	minimumMatchingNgrams := flag.Int("minimum_matching_ngrams", 4, "minimum matching ngrams to constitue a match")
	minimumMatchingNgramsInWindow := flag.Int("minimum_matching_ngrams_in_window", 4, "minimum matching ngrams per sliding window")
	minimumMatchingNgramsInDocs := flag.Int("minimum_matching_ngrams_in_docs", 4, "minimum unique ngrams matching between docs to start comparison")
	contextSize := flag.Int("context_size", 300, "size of context for before and after matching passages")
	banalNgrams := flag.Int("banal_ngrams", 25, "The top banal ngrams between two docs: used to define common, or banal ngrams")
	duplicateThreshold := flag.Int("duplicate_threshold", 80, "dismiss comparison if two texts share n or more percent of ngrams")
	mergeOnByteDistance := flag.Bool("merge_passages_on_byte_distance", true, "Merge passages within x number of byte: number defined by passage length and the passage_distance_multiplier option. Value between 0 and 1")
	mergeOnNgramDistance := flag.Bool("merge_passages_on_ngram_distance", true, "Merge passages within x number of ngrams: the value used is the matching_window_size defaulting to 20")
	passageDistance := flag.Float64("passage_distance_multiplier", 0.5, "Combine passage which are within (multiplier*length of previous passage) bytes")
	debugArg := flag.String("debug", "false", "set debugging: you need to also provide the --ngram_index option with a path to the ngram index to debug the matching logic.")
	flag.Parse()
	debug, _ := strconv.ParseBool(*debugArg)
	config := &matchingParams{int32(*matchingWindowSize), int32(*maxGap), *flexGap, int32(*minimumMatchingNgrams), int32(*minimumMatchingNgramsInWindow), float32(*commonNgramsLimit) / 100, *minimumMatchingNgramsInDocs,
		int32(*contextSize), *banalNgrams, *mergeOnByteDistance, *mergeOnNgramDistance, float64(*passageDistance), float64(*duplicateThreshold), *sourceBatch, *targetBatch, *outputPath, *threadsArg, *sortField, debug}
	ngramIndex := make(map[int32]string)
	if config.debug && *ngramIndexLocation != "" {
		ngramIndex = loadNgramIndex(*ngramIndexLocation)
	} else {
		ngramIndex = map[int32]string{}
	}
	fmt.Printf("Loading metadata...")
	if *sourceMetadataArg == "" {
		fmt.Println("\nNo source metadata provided, stopping now...")
		os.Exit(-1)
	}
	sourceMetadata := openJSONMetadata(sourceMetadataArg)
	targetMetadata := openJSONMetadata(targetMetadataArg)
	fmt.Println("done.")
	sourceFiles := getFiles(*sourceFilesArg, sourceMetadata, *sortField)
	if *targetFilesArg == *sourceFilesArg {
		*targetFilesArg = ""
	}
	targetFiles := getFiles(*targetFilesArg, targetMetadata, *sortField)
	if len(targetFiles) > 0 && *targetMetadataArg == "" {
		fmt.Println("\nNo target metadata provided, stopping now...")
		os.Exit(-1)
	}
	mostCommonNgrams := compileMostCommonNgrams(sourceCommonNgramsArg, targetCommonNgramsArg, mostCommonNgramThreshold)
	return sourceFiles, targetFiles, sourceMetadata, targetMetadata, mostCommonNgrams, config, ngramIndex
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
	checkErr(err, "openJSONMetadata")
	metadata := make(map[string]map[string]string)
	json.Unmarshal(jsonFile, &metadata)
	if len(metadata) == 0 {
		fmt.Printf("Metadata file %s is empty, stopping alignment...\n", *fileLocation)
		os.Exit(-1)
	}
	for doc, fields := range metadata {
		for field, value := range fields {
			metadata[doc][field] = spaceChars.ReplaceAllString(value, " ") // clean up metadata
		}
	}
	return metadata
}

func getFiles(filePath string, metadata map[string]map[string]string, sortField string) []sortedFile {
	if filePath == "" {
		return []sortedFile{}
	}
	if !strings.HasPrefix(filePath, "/") {
		filePath = "./" + filePath
	}
	directory, err := os.Open(filePath)
	checkErr(err, "getFiles (opening directory)")
	files, err := directory.Readdir(-1)
	checkErr(err, "getFiles (reading directory)")
	var filesToLoad []string
	for _, fileInfo := range files {
		if !fileInfo.IsDir() {
			file := filepath.Join(filePath, fileInfo.Name())
			filesToLoad = append(filesToLoad, file)
		}
	}
	sortFieldIsNumeric := false
	for _, fields := range metadata {
		if _, ok := fields[sortField]; !ok {
			sortField = ""
			break
		}
		if _, err := strconv.Atoi(fields[sortField]); err == nil {
			sortFieldIsNumeric = true
		}
		break
	}
	if sortField != "" {
		sort.Slice(filesToLoad, func(i, j int) bool {
			first := path.Base(strings.Replace(filesToLoad[i], ".json", "", 1))
			second := path.Base(strings.Replace(filesToLoad[j], ".json", "", 1))
			if sortFieldIsNumeric {
				firstInt, err := strconv.Atoi(metadata[first][sortField])
				if err != nil {
					return first < second
				}
				secondInt, err := strconv.Atoi(metadata[second][sortField])
				if err != nil {
					return first < second
				}
				if firstInt < secondInt {
					return true
				} else if firstInt > secondInt {
					return false
				}
				firstNameInt, _ := strconv.Atoi(first)
				secondNameInt, _ := strconv.Atoi(second)
				return firstNameInt < secondNameInt
			}
			return metadata[first][sortField] < metadata[second][sortField]
		})
	} else {
		sort.Slice(filesToLoad, func(i, j int) bool {
			first, _ := strconv.Atoi(path.Base(strings.Replace(filesToLoad[i], ".json", "", 1)))
			second, _ := strconv.Atoi(path.Base(strings.Replace(filesToLoad[j], ".json", "", 1)))
			return first < second
		})
	}
	sortedFilesToLoad := []sortedFile{}
	for pos, file := range filesToLoad {
		sortedFilesToLoad = append(sortedFilesToLoad, sortedFile{file, pos})
	}
	return sortedFilesToLoad
}

func getJSONDocs(fileLocations []sortedFile, prefixString string, threads int) []docIndex {
	var jsonFiles []docIndex
	totalFiles := len(fileLocations)
	runningTotal := 0
	multiplier := threads * 4
	var c chan docIndex
	groupNum := int(math.Floor(float64(totalFiles)/float64(multiplier))) + 1
	fileGroups := makeSliceOfSlices(fileLocations, groupNum)
	for _, fileGroup := range fileGroups {
		filesInGroup := len(fileGroup)
		c = make(chan docIndex, filesInGroup)
		var wait sync.WaitGroup
		wait.Add(filesInGroup)
		for _, fileLocation := range fileGroup {
			go func(fileLocation sortedFile) {
				defer wait.Done()
				jsonFile, err := ioutil.ReadFile(fileLocation.docID)
				checkErr(err, "getJSONDocs")
				tempDoc := make(map[int32][][]int32)
				json.Unmarshal(jsonFile, &tempDoc)
				doc := make(map[int32][]indexedNgram)
				for key, value := range tempDoc {
					doc[key] = []indexedNgram{}
					for _, ngram := range value {
						doc[key] = append(doc[key], indexedNgram{ngram[0], ngram[1], ngram[2]})
					}
				}
				docID := path.Base(strings.Replace(fileLocation.docID, ".json", "", 1))
				docObject := docIndex{docID, doc, len(doc), fileLocation.sortID}
				c <- docObject
			}(fileLocation)
		}
		wait.Wait()
		for i := 0; i < filesInGroup; i++ {
			localDocIndex := <-c
			jsonFiles = append(jsonFiles, localDocIndex)
			runningTotal++
		}
		progress := fmt.Sprintf("\r%s... %d/%d", prefixString, runningTotal, totalFiles)
		os.Stdout.Write([]byte(progress))
		os.Stdout.Sync()
	}
	os.Stdout.Write([]byte("\r\033[K" + prefixString + "... done.\n"))
	os.Stdout.Sync()
	sort.Slice(jsonFiles, func(i, j int) bool {
		return jsonFiles[i].SortID < jsonFiles[j].SortID
	})
	return jsonFiles
}

func compileMostCommonNgrams(sourceNgrams *string, targetNgrams *string, mostCommonNgramThreshold *int) map[int32]bool {
	uniqueNgrams := make(map[int32]bool)
	listOfFiles := []string{*sourceNgrams, *targetNgrams}
	for _, filename := range listOfFiles {
		if filename == "" {
			continue
		}
		file, err := os.Open(filename)
		defer file.Close()
		checkErr(err, "compileMostCommonNgrams")
		reader := bufio.NewReader(file)
		var line string
		for count := 0; count < *mostCommonNgramThreshold; count++ {
			line, err = reader.ReadString('\n')
			if err != nil {
				break
			}
			line = strings.TrimSpace(line)
			intNgram, _ := strconv.Atoi(line)
			uniqueNgrams[int32(intNgram)] = true
		}
	}
	return uniqueNgrams
}

func loadNgramIndex(fileLocation string) map[int32]string {
	file, err := os.Open(fileLocation)
	defer file.Close()
	checkErr(err, "loadNgramIndex")
	reader := bufio.NewReader(file)
	ngramIndex := make(map[int32]string)
	var line string
	for {
		line, err = reader.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		values := strings.Split(line, "\t")
		if len(values) == 2 { // avoid dying on empty line
			intValue, _ := strconv.Atoi(values[1])
			ngramIndex[int32(intValue)] = values[0]
		}
	}
	return ngramIndex
}

func alignPassages(sourceFiles []sortedFile, targetFiles []sortedFile, sourceMetadata map[string]map[string]string, targetMetadata map[string]map[string]string, commonNgrams map[int32]bool, config *matchingParams, ngramIndex map[int32]string) int {
	sourceAgainstSource := false

	// Split source and target files into config.batchSize batches
	if config.sourceBatch > len(sourceFiles) {
		config.sourceBatch = len(sourceFiles)
	}
	sourceFileBatches := makeSliceOfSlices(sourceFiles, config.sourceBatch)
	var targetFileBatches [][]sortedFile
	if len(targetFiles) == 0 {
		targetMetadata = sourceMetadata
		sourceAgainstSource = true
		targetFileBatches = sourceFileBatches
		config.targetBatch = config.sourceBatch
	} else {
		if config.targetBatch > len(targetFiles) {
			config.targetBatch = len(targetFiles)
		}
		targetFileBatches = makeSliceOfSlices(targetFiles, config.targetBatch)
	}
	mergedOutput := createOutputFile(config)
	duplicateFilesOutput := creatDuplicateFilesOutputFile(config)
	counts := 0
	for sourceBatchNumber := 0; sourceBatchNumber < config.sourceBatch; sourceBatchNumber++ {
		prefixString := "Loading source files"
		if config.sourceBatch > 1 {
			prefixString += fmt.Sprintf(" from source batch %d", sourceBatchNumber+1)
			fmt.Printf("\n### Comparing source batch %d against all... ###\n", sourceBatchNumber+1)
		}
		sourceFileIndexes := getJSONDocs(sourceFileBatches[sourceBatchNumber], prefixString, config.numThreads)
		for targetBatchNumber := 0; targetBatchNumber < config.targetBatch; targetBatchNumber++ {
			if sourceAgainstSource && sourceBatchNumber > targetBatchNumber {
				continue // we've already done these comparisons in the other direction
			}
			var targetFileIndexes []docIndex
			if sourceAgainstSource && targetBatchNumber == sourceBatchNumber {
				targetFileIndexes = sourceFileIndexes
			} else {
				targetPrefix := "Loading target files"
				if config.targetBatch > 1 {
					targetPrefix += fmt.Sprintf(" from target batch %d", targetBatchNumber+1)
				}
				targetFileIndexes = getJSONDocs(targetFileBatches[targetBatchNumber], targetPrefix, config.numThreads)
			}
			percentSteps := buildPercentMap(len(sourceFileIndexes))
			fmt.Printf("Comparing files... 0%%")
			for pos, sourceFile := range sourceFileIndexes {
				if config.debug {
					if config.sourceBatch == 1 {
						fmt.Printf("Comparing source file %s to all...\n", sourceFile.DocID)
					} else {
						fmt.Printf("Comparing source file %s to target batch %d...\n", sourceFile.DocID, targetBatchNumber+1)
					}
				}
				if _, ok := percentSteps[pos]; ok {
					percent := strconv.Itoa(percentSteps[pos])
					os.Stdout.Write([]byte("\rComparing files... " + percent + "%"))
					os.Stdout.Sync()
				}
				var wait sync.WaitGroup
				targetLength := len(targetFileIndexes)

				var start int
				if sourceAgainstSource && sourceBatchNumber == targetBatchNumber {
					start = pos + 1
				} else {
					start = 0
				}
				var increment int
				threadsNeeded := config.numThreads
				if config.numThreads > 1 {
					localTargetLength := len(targetFileIndexes[start:])
					filesPerThread := localTargetLength / threadsNeeded
					for filesPerThread < 10 {
						threadsNeeded = threadsNeeded / 2 // We reduce the number of Go routines to avoid starvation.
						if threadsNeeded < 2 {
							threadsNeeded = 1
							break
						}
						filesPerThread = localTargetLength / threadsNeeded
					}
					increment = (targetLength-start)/threadsNeeded + 1
				} else {
					increment = targetLength - start
				}

				c := make(chan []alignmentsPerDoc, threadsNeeded)

				// Defer the closing of c to guarantee against unclosed channel and thus memory leaks
				defer close(c)

				wait.Add(threadsNeeded)
				end := start + increment
				totalTexts := 0
				for i := 0; i < threadsNeeded; i++ {
					if end > targetLength {
						end = targetLength
					}
					splitTargets := targetFileIndexes[start:end]
					totalTexts += len(splitTargets)
					start = end
					end += increment
					go func(splitTargets []docIndex, sourceAgainstSource bool, sourceMetadata map[string]map[string]string, targetMetadata map[string]map[string]string, config *matchingParams, commonNgrams map[int32]bool) {
						defer wait.Done()
						localAlignments := []alignmentsPerDoc{}
						for _, targetFile := range splitTargets {
							if sourceAgainstSource && sourceFile.SortID >= targetFile.SortID {
								continue
							}
							var debugOutput *os.File
							if config.debug {
								debugOutput = createDebugOutputFile(config, sourceFile.DocID, targetFile.DocID)
							}
							sourceTargetIntersection, totalCommonNgrams := getIntersection(&sourceFile, &targetFile)
							if len(sourceTargetIntersection) < config.minimumMatchingNgramsInDocs {
								continue
							} else if float64(totalCommonNgrams)/float64(sourceFile.NgramLength)*100 > config.duplicateThreshold {
								sourceInfo := fmt.Sprintf("%s\t%s\t%s\t%s\t%s-%s", sourceMetadata[sourceFile.DocID]["title"], sourceMetadata[sourceFile.DocID]["author"], sourceMetadata[sourceFile.DocID]["filename"], sourceMetadata[sourceFile.DocID]["philo_id"], sourceMetadata[sourceFile.DocID]["start_byte"], sourceMetadata[sourceFile.DocID]["end_byte"])
								targetInfo := fmt.Sprintf("%s\t%s\t%s\t%s\t%s-%s", targetMetadata[targetFile.DocID]["title"], targetMetadata[targetFile.DocID]["author"], targetMetadata[targetFile.DocID]["filename"], targetMetadata[targetFile.DocID]["philo_id"], targetMetadata[targetFile.DocID]["start_byte"], targetMetadata[targetFile.DocID]["end_byte"])
								duplicateInfo := fmt.Sprintf("%s\t%s\t%f", sourceInfo, targetInfo, float64(totalCommonNgrams)/float64(sourceFile.NgramLength)*100)
								localAlignments = append(localAlignments, alignmentsPerDoc{targetFile.DocID, []Alignment{}, duplicateInfo})
								continue
							}
							mostCommonNgrams := getMostCommonNgrams(sourceTargetIntersection, &config.banalNgrams, commonNgrams)
							var matches = []ngramMatch{}
							for ngram := range sourceTargetIntersection {
								for _, sourceNgramIndex := range sourceFile.Ngrams[ngram] {
									for _, targetNgramIndex := range targetFile.Ngrams[ngram] {
										matches = append(matches, ngramMatch{sourceNgramIndex, targetNgramIndex, ngram})
									}
								}
							}
							sort.Slice(matches, func(i, j int) bool {
								if matches[i].source.index < matches[j].source.index {
									return true
								} else if matches[i].source.index > matches[j].source.index {
									return false
								}
								return matches[i].target.index < matches[j].target.index
							})
							alignments := matchPassage(&sourceFile, &targetFile, matches, config, mostCommonNgrams, ngramIndex, debugOutput)
							if config.mergeOnByteDistance || config.mergeOnNgramDistance {
								alignments = mergeWithPrevious(alignments, config, debugOutput)
							}
							if len(alignments) > 0 {
								localAlignments = append(localAlignments, alignmentsPerDoc{targetFile.DocID, alignments, ""})
							}
							debugOutput.Sync()
							debugOutput.Close()
						}
						c <- localAlignments
					}(splitTargets, sourceAgainstSource, sourceMetadata, targetMetadata, config, commonNgrams)
				}
				wait.Wait()
				for i := 0; i < threadsNeeded; i++ {
					localCombinedAlignments := <-c
					if len(localCombinedAlignments) > 0 {
						writeAlignments(localCombinedAlignments, &sourceFile.DocID, sourceMetadata, targetMetadata, mergedOutput, duplicateFilesOutput, config, &counts)
					}
				}
			}
			os.Stdout.Write([]byte("\r\033[KComparing files... done.\n"))
			os.Stdout.Sync()
		}
	}
	mergedOutput.Sync()
	mergedOutput.Close()
	fmt.Printf("%d pairwise alignments found...\n", counts)
	return counts
}

func getIntersection(sourceFile *docIndex, targetFile *docIndex) (map[int32]int, int) {
	intersectCount := make(map[int32]int)
	totalCommonNgrams := 0
	if sourceFile.NgramLength < targetFile.NgramLength {
		for ngram := range sourceFile.Ngrams {
			if _, ok := targetFile.Ngrams[ngram]; ok {
				intersectCount[ngram] = len(sourceFile.Ngrams[ngram]) + len(targetFile.Ngrams[ngram])
				totalCommonNgrams++
			}
		}
	} else {
		for ngram := range targetFile.Ngrams {
			if _, ok := sourceFile.Ngrams[ngram]; ok {
				intersectCount[ngram] = len(sourceFile.Ngrams[ngram]) + len(targetFile.Ngrams[ngram])
				totalCommonNgrams++
			}
		}
	}
	return intersectCount, totalCommonNgrams
}

func getMostCommonNgrams(intersectionCount map[int32]int, banalNgrams *int, commonNgrams map[int32]bool) map[int32]bool {
	sortedIntersection := sortMapByValue(intersectionCount)
	mostCommonNgrams := make(map[int32]bool, len(commonNgrams))
	var count int
	for _, pair := range sortedIntersection {
		if pair.Value == 2 {
			break
		}
		mostCommonNgrams[pair.Key] = true
		count++
		if count == *banalNgrams {
			break
		}
	}
	for commonNgrams := range commonNgrams {
		mostCommonNgrams[commonNgrams] = true
	}
	return mostCommonNgrams
}

func createOutputFile(config *matchingParams) *os.File {
	os.MkdirAll(config.outputPath, 0755)
	configOutput, err := os.Create(filepath.Join(config.outputPath, "alignment_config.ini"))
	configOutput.WriteString("## Alignment Parameters ##\n\n")
	matchingParameters := []string{
		"matchingWindowSize",
		"maxGap",
		"flexGap",
		"minimumMatchingNgrams",
		"minimumMatchingNgramsInWindow",
		"commonNgramsLimit",
		"minimumMatchingNgramsInDocs",
		"contextSize",
		"banalNgrams",
		"mergeOnByteDistance",
		"mergeOnNgramDistance",
		"passageDistanceMultiplier",
		"oneWayMatching",
		"duplicateThreshold",
		"sourceBatch",
		"targetBatch",
		"outputPath",
		"numThreads",
		"sortingField",
		"debug",
	}
	v := reflect.ValueOf(*config)
	for _, param := range matchingParameters {
		f := reflect.Indirect(v).FieldByName(param)
		configOutput.WriteString(fmt.Sprintf("%s: %v\n", param, f))
	}
	configOutput.Sync()
	configOutput.Close()

	mergedOutput, err := os.Create(filepath.Join(config.outputPath, "alignments.jsonl"))
	checkErr(err, "createOutputFile")
	return mergedOutput
}

func createDebugOutputFile(config *matchingParams, sourceDocID string, targetDocID string) *os.File {
	debugOutputPath := fmt.Sprintf(filepath.Join(config.outputPath, "debug_output"))
	if _, err := os.Stat(debugOutputPath); os.IsNotExist(err) {
		os.MkdirAll(debugOutputPath, 0755)
	}
	outputFile := fmt.Sprintf("%s_%s", sourceDocID, targetDocID)
	debugOutput, err := os.Create(filepath.Join(debugOutputPath, outputFile))
	checkErr(err, "createDebugOutputFile")
	return debugOutput
}

func creatDuplicateFilesOutputFile(config *matchingParams) *os.File {
	duplicateFiles, err := os.Create(fmt.Sprintf(filepath.Join(config.outputPath, "duplicate_files.tsv")))
	checkErr(err, "creatDuplicateFilesOutputFile")
	duplicateFiles.WriteString("source_title\tsource_author\tsource_filename\tsource_philo_id\tsource_byte_offsets\ttarget_title\ttarget_author\ttarget_filename\target_philo_id\ttarget_byte_offsets\toverlap\n")
	return duplicateFiles
}

func matchPassage(sourceFile *docIndex, targetFile *docIndex, matches []ngramMatch, config *matchingParams, mostCommonNgrams map[int32]bool, ngramIndex map[int32]string, debugOutput *os.File) []Alignment {
	alignments := make([]Alignment, 0)
	m := &matchValues{}
	m.lastSourcePosition = 0
	m.inAlignment = false
	for matchIndex, currentAnchor := range matches {
		if currentAnchor.source.index < m.lastSourcePosition {
			continue
		}
		m.sourceAnchor = currentAnchor.source.index
		m.sourceWindowBoundary = m.sourceAnchor + config.matchingWindowSize
		m.lastSourcePosition = m.sourceAnchor
		m.maxSourceGap = m.lastSourcePosition + config.maxGap
		m.targetAnchor = currentAnchor.target.index
		m.targetWindowBoundary = m.targetAnchor + config.matchingWindowSize
		m.lastTargetPosition = m.targetAnchor
		m.maxTargetGap = m.lastTargetPosition + config.maxGap
		m.inAlignment = true
		m.previousSourceIndex = m.sourceAnchor
		m.firstMatch = []indexedNgram{currentAnchor.source, currentAnchor.target}
		m.matchesInCurrentAlignment = 1
		m.matchesInCurrentWindow = 1
		m.commonNgramMatches = 0
		if _, ok := mostCommonNgrams[currentAnchor.ngram]; ok {
			m.commonNgramMatches++
		}
		m.lastMatch = []indexedNgram{currentAnchor.source, currentAnchor.target}
		if config.debug {
			m.debug = []string{ngramIndex[currentAnchor.ngram]}
		}
		currentMatchesLength := len(matches)
		maxGap := config.maxGap
		matchingWindowSize := config.matchingWindowSize
	innerMatchingLoop:
		for pos, match := range matches[matchIndex+1:] {
			source, target := match.source, match.target
			// we skip source_match if the same as before and we only want targets that are after last target match
			if source.index == m.previousSourceIndex {
				continue
			}
			if target.index > m.maxTargetGap || target.index <= m.lastTargetPosition {
				nextIndex := pos + matchIndex + 1
				// Is next source index within maxSourceGap? If so, the match should continue since target may be within maxTargetGap
				if nextIndex <= currentMatchesLength && matches[nextIndex].source.index <= m.maxSourceGap {
					continue
				} else {
					m.inAlignment = false
				}
			}
			if source.index > m.maxSourceGap && m.matchesInCurrentWindow < config.minimumMatchingNgramsInWindow {
				m.inAlignment = false
			}
			if source.index > m.sourceWindowBoundary || target.index > m.targetWindowBoundary {
				if m.matchesInCurrentWindow < config.minimumMatchingNgramsInWindow {
					m.inAlignment = false
				} else {
					if source.index > m.maxSourceGap || target.index > m.maxTargetGap {
						m.inAlignment = false
					} else {
						m.sourceAnchor = source.index
						m.sourceWindowBoundary = m.sourceAnchor + matchingWindowSize
						m.targetAnchor = target.index
						m.targetWindowBoundary = m.targetAnchor + matchingWindowSize
						m.matchesInCurrentWindow = 0
					}
				}
			}
			if !m.inAlignment {
				if m.matchesInCurrentAlignment >= config.minimumMatchingNgrams {
					addAlignment(m, config, &alignments)
					if config.debug {
						writeDebugOutput(m, true, &currentAnchor, debugOutput)
					}
				} else if config.debug {
					writeDebugOutput(m, false, &currentAnchor, debugOutput)
				}
				m.lastSourcePosition = m.lastMatch[0].index + 1 // Make sure we start the next match at index that follows last source match
				break innerMatchingLoop
			}
			m.lastSourcePosition = source.index
			m.maxSourceGap = m.lastSourcePosition + maxGap
			m.lastTargetPosition = target.index
			m.maxTargetGap = m.lastTargetPosition + maxGap
			m.previousSourceIndex = source.index
			m.matchesInCurrentWindow++
			m.matchesInCurrentAlignment++
			if config.flexGap {
				if m.matchesInCurrentAlignment == config.minimumMatchingNgrams {
					maxGap += config.minimumMatchingNgrams
					matchingWindowSize += config.minimumMatchingNgrams
				} else if m.matchesInCurrentAlignment > config.minimumMatchingNgrams {
					if maxGap < config.matchingWindowSize { // Gaps should not go beyond initial window size: prevents huge jumps with one ngram at the end
						maxGap++
						matchingWindowSize++
					}
				}
			}
			m.lastMatch = []indexedNgram{source, target} // save last matching ngrams
			if _, ok := mostCommonNgrams[match.ngram]; ok {
				m.commonNgramMatches++
			}
			if config.debug {
				m.debug = append(m.debug, ngramIndex[match.ngram])
			}
		}
		if m.inAlignment && m.matchesInCurrentAlignment >= config.minimumMatchingNgrams {
			addAlignment(m, config, &alignments)
		}
	}
	return alignments
}

// Merge alignments based on either byte distance or ngram distance
func mergeWithPrevious(alignments []Alignment, config *matchingParams, debugOutput *os.File) []Alignment {
	var maxSourceDistance, maxTargetDistance int32
	var maxNgramDistance int32
	maxSourceDistance = 0
	maxTargetDistance = 0
	if config.mergeOnNgramDistance {
		maxNgramDistance = config.matchingWindowSize
	} else {
		maxNgramDistance = 0
	}
	var mergedAlignments []Alignment
	var previousAlignment Alignment
	lastIndex := len(alignments) - 1
	for index, currentAlignment := range alignments { // This code assumes that alignments are sorted, with source first, then target
		if index == 0 {
			previousAlignment = currentAlignment
			continue
		}
		currentAlignmentMerged := false
		if config.mergeOnByteDistance {
			distanceValue := int32(math.Floor((float64(previousAlignment.source.endByte - previousAlignment.source.startByte)) * config.passageDistanceMultiplier))
			maxSourceDistance = previousAlignment.source.endByte + distanceValue
			maxTargetDistance = previousAlignment.target.endByte + distanceValue
		}
		sourceNgramDistance := previousAlignment.source.endNgramIndex + maxNgramDistance
		targetNgramDistance := previousAlignment.target.endNgramIndex + maxNgramDistance

		if currentAlignment.source.startByte <= maxSourceDistance &&
			currentAlignment.target.startByte <= maxTargetDistance &&
			currentAlignment.target.startByte > previousAlignment.target.endByte {
			currentAlignmentMerged = true
			sourcePosition := position{previousAlignment.source.startByte, currentAlignment.source.endByte, previousAlignment.source.startNgramIndex, currentAlignment.source.endNgramIndex}
			targetPosition := position{previousAlignment.target.startByte, currentAlignment.target.endByte, previousAlignment.target.startNgramIndex, currentAlignment.target.endNgramIndex}
			previousAlignment = Alignment{sourcePosition, targetPosition, previousAlignment.totalMatchingNgrams + currentAlignment.totalMatchingNgrams, false} // we consider merged passages as non-banality
		} else if currentAlignment.source.startNgramIndex <= sourceNgramDistance &&
			currentAlignment.target.startNgramIndex <= targetNgramDistance &&
			currentAlignment.target.startNgramIndex > previousAlignment.target.endNgramIndex {
			currentAlignmentMerged = true
			sourcePosition := position{previousAlignment.source.startByte, currentAlignment.source.endByte, previousAlignment.source.startNgramIndex, currentAlignment.source.endNgramIndex}
			targetPosition := position{previousAlignment.target.startByte, currentAlignment.target.endByte, previousAlignment.target.startNgramIndex, currentAlignment.target.endNgramIndex}
			previousAlignment = Alignment{sourcePosition, targetPosition, previousAlignment.totalMatchingNgrams + currentAlignment.totalMatchingNgrams, false} // we consider merged passages as non-banality
		} else {
			mergedAlignments = append(mergedAlignments, previousAlignment) // we store previous since it can no longer be merged with next
			previousAlignment = currentAlignment                           // current match was not merged with previous so now becomes previous
		}
		if index == lastIndex { // don't forget to add last unmerged alignment
			if currentAlignmentMerged {
				mergedAlignments = append(mergedAlignments, previousAlignment)
			} else {
				mergedAlignments = append(mergedAlignments, currentAlignment)
			}
		}
	}
	if (Alignment{}) != previousAlignment && len(mergedAlignments) == 0 {
		mergedAlignments = append(mergedAlignments, previousAlignment)
	}
	if config.debug && len(alignments) > len(mergedAlignments) {
		debugOutput.WriteString(fmt.Sprintf("\n\n%d passage(s) merged with previous passage", len(alignments)-len(mergedAlignments)))
		debugOutput.Sync()
	}
	return mergedAlignments
}

func writeDebugOutput(m *matchValues, match bool, currentAnchor *ngramMatch, debugOutput *os.File) {
	var stringOutput string
	if match {
		stringOutput = "\n\n## MATCH ##\n"
		stringOutput += fmt.Sprintf("Source byte range: %d-%d\n", m.currentAlignment.source.startByte, m.currentAlignment.source.endByte)
		stringOutput += fmt.Sprintf("Source matching index range: %d-%d\n", m.firstMatch[0].index, m.lastMatch[0].index)
		stringOutput += fmt.Sprintf("Target byte range: %d-%d\n", m.currentAlignment.target.startByte, m.currentAlignment.target.endByte)
		stringOutput += fmt.Sprintf("Target matching index range: %d-%d\n", m.firstMatch[1].index, m.lastMatch[1].index)
		stringOutput += fmt.Sprintf("Matching ngrams: %s", strings.Join(m.debug, " "))
		stringOutput += fmt.Sprintf("Number of matching ngrams: %d", len(m.debug))
	} else {
		stringOutput = "\n\n## FAILED MATCH ##\n"
		stringOutput += fmt.Sprintf("Source byte range: %d-%d\n", m.firstMatch[0].startByte, m.lastMatch[0].endByte)
		stringOutput += fmt.Sprintf("Source matching index range: %d-%d\n", m.firstMatch[0].index, m.lastMatch[0].index)
		stringOutput += fmt.Sprintf("Target byte range: %d-%d\n", m.firstMatch[1].startByte, m.lastMatch[1].endByte)
		stringOutput += fmt.Sprintf("Target matching index range: %d-%d\n", m.firstMatch[1].index, m.lastMatch[1].index)
		stringOutput += fmt.Sprintf("Matching ngrams: %s\n", strings.Join(m.debug, " "))
		stringOutput += fmt.Sprintf("Number of matching ngrams: %d", len(m.debug))
	}
	debugOutput.WriteString(stringOutput)
	debugOutput.Sync()
}

func writeAlignments(localAlignments []alignmentsPerDoc, sourceDocID *string, sourceMetadata map[string]map[string]string,
	targetMetadata map[string]map[string]string, f *os.File, duplicatesFile *os.File, config *matchingParams, counts *int) {
	for _, alignments := range localAlignments {
		fullAlignment := map[string]string{}
		for key, value := range sourceMetadata[*sourceDocID] {
			fullAlignment["source_"+key] = value
		}
		for key, value := range targetMetadata[alignments.docID] {
			fullAlignment["target_"+key] = value
		}
		fullAlignment["source_doc_id"] = *sourceDocID
		fullAlignment["target_doc_id"] = alignments.docID
		for _, alignment := range alignments.matches {
			localAlignment := fullAlignment
			localAlignment["source_start_byte"] = strconv.Itoa(int(alignment.source.startByte))
			localAlignment["source_end_byte"] = strconv.Itoa(int(alignment.source.endByte))
			sourcePassages := alignmentToText(&alignment.source, sourceMetadata[*sourceDocID]["filename"], config)
			localAlignment["source_context_before"] = sourcePassages[0]
			localAlignment["source_passage"] = sourcePassages[1]
			localAlignment["source_context_after"] = sourcePassages[2]
			// sourcePositions := getRelativePosition(alignment.source.startByte, alignment.source.endByte, sourceMetadata, sourceDocID)
			// localAlignment["source_start_position"] = sourcePositions[0]
			// localAlignment["source_end_position"] = sourcePositions[1]
			localAlignment["target_start_byte"] = strconv.Itoa(int(alignment.target.startByte))
			localAlignment["target_end_byte"] = strconv.Itoa(int(alignment.target.endByte))
			targetPassages := alignmentToText(&alignment.target, targetMetadata[alignments.docID]["filename"], config)
			localAlignment["target_context_before"] = targetPassages[0]
			localAlignment["target_passage"] = targetPassages[1]
			localAlignment["target_context_after"] = targetPassages[2]
			// targetPositions := getRelativePosition(alignment.target.startByte, alignment.target.endByte, targetMetadata, &alignments.docID)
			// localAlignment["target_start_position"] = targetPositions[0]
			// localAlignment["target_end_position"] = targetPositions[1]
			localAlignment["banality"] = fmt.Sprintf("%v", alignment.banality)
			*counts++
			localAlignment["passage_id"] = strconv.Itoa(*counts)
			jsonString, _ := json.Marshal(localAlignment)
			jsonString = append(jsonString, "\n"...)
			f.Write(jsonString)
		}
		if len(alignments.duplicates) > 0 {
			duplicatesFile.WriteString(fmt.Sprintf("%s\n", alignments.duplicates))
		}
	}
}

// Returns position of passage within document in percentages
func getRelativePosition(startByte int32, endByte int32, metadata map[string]map[string]string, docID *string) []string {
	docStart, _ := strconv.Atoi(metadata[*docID]["start_byte"])
	docEnd, _ := strconv.Atoi(metadata[*docID]["end_byte"])
	coefficient := (float64(docEnd) - float64(docStart)) / 100
	startPosition := fmt.Sprintf("%d", int(math.Round(float64(startByte)/coefficient)))
	endPosition := fmt.Sprintf("%d", int(math.Round(float64(endByte)/coefficient)))
	return []string{startPosition, endPosition}
}

// Returns three passages: the context before, the match itself, and the context after
func alignmentToText(alignment *position, filename string, config *matchingParams) []string {
	beforeContext := getText(&filename, alignment.startByte-int32(config.contextSize), alignment.startByte, "before")
	beforeContext = cleanStart.ReplaceAllString(beforeContext, "") // avoid truncation at beginning
	matchingPassage := getText(&filename, alignment.startByte, alignment.endByte, "match")
	afterContext := getText(&filename, alignment.endByte, alignment.endByte+int32(config.contextSize), "after")
	afterContext = cleanEnd.ReplaceAllString(afterContext, "") // avoid truncation at the end
	passages := []string{beforeContext, matchingPassage, afterContext}
	return passages
}

// Get text passages using file location and start and end byte
func getText(fileLocation *string, startByte int32, endByte int32, passageType string) string {
	f, err := os.Open(*fileLocation)
	checkErr(err, fmt.Sprintf("getText (opening %s)", *fileLocation))
	if startByte < 0 {
		startByte = int32(0)
	}
	_, err = f.Seek(int64(startByte), 0)
	checkErr(err, "getText (seeking in file)")
	passage := make([]byte, endByte-startByte)
	_, err = f.Read(passage)
	checkErr(err, "getText (reading in file)")
	f.Close()
	passage = bytes.Trim(passage, "\x00")
	passage = bytes.Replace(passage, []byte("\xc2\xa0"), []byte(" "), -1) // remove non-breaking spaces
	text := string(passage)
	text = tags.ReplaceAllString(text, "")
	if passageType != "match" { // avoid truncating matching passage
		text = brokenBeginTags.ReplaceAllString(text, "")
		text = brokenEndTags.ReplaceAllString(text, "")
	}
	text = html.UnescapeString(text)
	text = strings.Replace(text, "\\n", "\n", -1)
	text = strings.Replace(text, "\\t", "\t", -1)
	text = strings.Replace(text, "\\r", "\r", -1)
	text = strings.Replace(text, "\t", " ", -1)
	text = tabEntities.ReplaceAllString(text, " ")
	text = strings.Replace(text, "\n", " ", -1)
	text = spaces.ReplaceAllString(text, " ")
	return text
}

// Add alignments to list of alignments
func addAlignment(m *matchValues, config *matchingParams, alignments *[]Alignment) {
	m.currentAlignment.source = position{m.firstMatch[0].startByte, m.lastMatch[0].endByte, m.firstMatch[0].index, m.lastMatch[0].index}
	m.currentAlignment.target = position{m.firstMatch[1].startByte, m.lastMatch[1].endByte, m.firstMatch[1].index, m.lastMatch[1].index}
	m.currentAlignment.totalMatchingNgrams = m.matchesInCurrentAlignment
	if float32(m.commonNgramMatches/m.matchesInCurrentAlignment) >= config.commonNgramsLimit {
		m.currentAlignment.banality = true
	}
	*alignments = append(*alignments, m.currentAlignment)
	m.previousAlignment = m.currentAlignment
}

// Helper functions
func buildPercentMap(total int) map[int]int {
	percentSteps := make(map[int]int)
	count := 0
	totalFloat := float64(total)
	step := totalFloat / 100.0
	for i := step; i < totalFloat; i += step {
		count++
		floor := int(math.Floor(i))
		percentSteps[floor] = count
	}
	return percentSteps
}

func checkErr(err error, errorMessage string) {
	if err != nil {
		fmt.Printf("An error occured in following function %s. See error message below:\n", errorMessage)
		log.Fatal(err)
	}
}

func makeSliceOfSlices(sliceToSlice []sortedFile, batch int) [][]sortedFile {
	var sliceOfSlices [][]sortedFile
	sliceLength := len(sliceToSlice)
	chunkSize := (sliceLength + batch - 1) / batch
	for i := 0; i < sliceLength; i += chunkSize {
		end := i + chunkSize
		if end > sliceLength {
			end = sliceLength
		}
		sliceOfSlices = append(sliceOfSlices, sliceToSlice[i:end])
	}
	return sliceOfSlices
}

func mapToSliceOfKeys(metadata map[string]string) []string {
	keys := []string{}
	for k := range metadata {
		keys = append(keys, k)
	}
	return keys
}

func mapToSliceOfValues(metadata map[string]string, fields []string) []string {
	values := []string{}
	for _, v := range fields {
		values = append(values, metadata[v])
	}
	return values
}
