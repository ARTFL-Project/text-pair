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
	minimumMatchingNgramsInDocs   int32
	contextSize                   int32
	banalNgrams                   int32
	batchSteps                    int
	outputPath                    string
	numThreads                    int
	outputFormat                  string
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

func parseFlags() ([]docIndex, []docIndex, map[string]map[string]string, map[string]map[string]string, matchingParams) {
	sourceFilesArg := flag.String("source_files", "", "source files location")
	targetFilesArg := flag.String("target_files", "", "target files location")
	threadsArg := flag.Int("threads", 4, "number of threads to use")
	sourceMetadataArg := flag.String("source_metadata", "", "path to source metadata")
	targetMetadataArg := flag.String("target_metadata", "", "path to target metadata")
	batchSteps := flag.Int("batch_steps", 1, "batch steps defines the number of steps in which the full source vs target is run: useful when RAM usage is a concern")
	matchingWindowSize := int32(*flag.Int("matching_window_size", 20, "size of sliding window for matches"))
	maxGap := int32(*flag.Int("max_gap", 10, "maximum gap between two matching ngrams"))
	minimumMatchingNgrams := int32(*flag.Int("minimum_matching_ngrams", 4, "minimum matching ngrams to constitue a match"))
	minimumMatchingNgramsInWindow := int32(*flag.Int("minimum_matching_ngram_in_window", 4, "minimum matching ngrams per sliding window"))
	commonNgramsLimit := float32(*flag.Int("common_ngrams_limit", 75, "percentage of common ngrams to dismiss a match as banal")) / 100
	minimumMatchingNgramsInDocs := int32(*flag.Int("minimum_matching_ngrams_in_docs", 4, "minimum unique ngrams matching between docs to start comparison"))
	contextSize := int32(*flag.Int("context_size", 300, "size of context for before and after matching passages"))
	banalNgrams := int32(*flag.Int("banal_ngrams", 25, "The top banal ngrams between two docs: used to define common, or banal ngrams"))
	outputPath := flag.String("output_path", "./", "output path for results")
	outputFormat := flag.String("output_format", "tab", "output format of results")
	flag.Parse()
	config := matchingParams{matchingWindowSize, maxGap, minimumMatchingNgrams, minimumMatchingNgramsInWindow, commonNgramsLimit, minimumMatchingNgramsInDocs,
		contextSize, banalNgrams, *batchSteps, *outputPath, *threadsArg, *outputFormat}
	fmt.Println("Reading JSON files...")
	sourceFiles := getFiles(*sourceFilesArg)
	targetFiles := getFiles(*targetFilesArg)
	fmt.Println("Loading metadata...")
	sourceMetadata := openJSONMetadata(sourceMetadataArg)
	targetMetadata := openJSONMetadata(targetMetadataArg)
	return sourceFiles, targetFiles, sourceMetadata, targetMetadata, config
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

func getMostCommonNgrams(intersectionCount map[int32]int, banalNgrams *int32) map[int32]bool {
	sortedIntersection := sortMapByValue(intersectionCount)
	mostCommonNgrams := make(map[int32]bool)
	var count int32
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
	return mostCommonNgrams
}

func writeAligments(combinedAlignments *CombinedAlignments, sourceDocID *string, sourceMetadata map[string]map[string]string,
	targetMetadata map[string]map[string]string, f *os.File, config *matchingParams) {
	var combinedOutput []string
	_, sourceValues := mapToSliceOfSlices(sourceMetadata[*sourceDocID])
	for _, alignments := range combinedAlignments.alignments {
		_, targetValues := mapToSliceOfSlices(targetMetadata[alignments.docID])
		// matchOutputs := []string{}
		for _, alignment := range alignments.matches {
			fields := []string{}
			fields = append(fields, sourceValues...)
			fields = append(fields, []string{strconv.Itoa(int(alignment.source.startByte)), strconv.Itoa(int(alignment.source.endByte))}...)
			fields = append(fields, alignmentToText(&alignment.source, sourceMetadata[*sourceDocID]["filename"], config)...)
			fields = append(fields, targetValues...)
			fields = append(fields, []string{strconv.Itoa(int(alignment.target.startByte)), strconv.Itoa(int(alignment.target.endByte))}...)
			fields = append(fields, alignmentToText(&alignment.target, targetMetadata[alignments.docID]["filename"], config)...)
			// matchOutputs = append(matchOutputs, strings.Join(fields, "\t"))
			combinedOutput = append(combinedOutput, strings.Join(fields, "\t"))
		}
	}
	f.WriteString("\n" + strings.Join(combinedOutput, "\n"))
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

func alignmentToText(alignment *position, filename string, config *matchingParams) []string {
	beforeContext := getText(&filename, alignment.startByte-int32(config.contextSize), alignment.startByte)
	matchingPassage := getText(&filename, alignment.startByte, alignment.endByte)
	afterContext := getText(&filename, alignment.endByte, alignment.endByte+int32(config.contextSize))
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

func createOutputFile(config *matchingParams, sourceMetadata map[string]map[string]string, targetMetadata map[string]map[string]string) *os.File {
	os.MkdirAll(config.outputPath, 0755)
	mergedOutput, err := os.Create(fmt.Sprintf(filepath.Join(config.outputPath, "alignments_results.%s"), config.outputFormat))
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
	sourceFiles, targetFiles, sourceMetadata, targetMetadata, config := parseFlags()
	sourceAgainstSource := false
	sourceFileDone := make(map[string]bool)
	if len(targetFiles) == 0 {
		targetFiles = sourceFiles
		targetMetadata = sourceMetadata
		sourceAgainstSource = true
	}
	mergedOutput := createOutputFile(&config, sourceMetadata, targetMetadata)

	counts := 0
	targetLength := len(targetFiles)
	var wait sync.WaitGroup
	for _, sourceFile := range sourceFiles {
		fmt.Printf("Comparing source file %s to all...\n", sourceFile.DocID)
		combinedAlignments := CombinedAlignments{sourceFile.DocID, []alignmentsPerDoc{}}
		c := make(chan []alignmentsPerDoc, config.numThreads)
		wait.Add(config.numThreads)
		start := 0
		increment := targetLength/config.numThreads + 1
		for i := 0; i < config.numThreads; i++ {
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
					if int32(len(sourceTargetIntersection)) < config.minimumMatchingNgramsInDocs {
						continue
					}
					mostCommonNgrams := getMostCommonNgrams(sourceTargetIntersection, &config.banalNgrams)
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
					m := matchValues{}
					m.lastSourcePosition = 0
					m.inAlignment = false
					for matchIndex, currentAnchor := range matches {
						if currentAnchor.source[0] < m.lastSourcePosition {
							continue
						}
						m.sourceAnchor = currentAnchor.source[0]
						m.sourceWindowBoundary = m.sourceAnchor + config.matchingWindowSize
						m.lastSourcePosition = m.sourceAnchor
						m.maxSourceGap = m.lastSourcePosition + config.maxGap
						m.targetAnchor = currentAnchor.target[0]
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
					innerMatchingLoop:
						for _, match := range matches[matchIndex+1:] {
							source, target := match.source, match.target
							// we skip source_match if the same as before and we only want targets that are after last target match
							if source[0] == m.previousSourceIndex || target[0] <= m.lastTargetPosition {
								continue
							}
							if source[0] > m.maxSourceGap || target[0] > m.maxTargetGap {
								// println("Failed", source[0], m.lastSourcePosition+matchingDefaults.maxGap, target[0], m.lastTargetPosition+matchingDefaults.maxGap)
								// println(firstStep, secondStep)
								m.inAlignment = false
							}
							if source[0] > m.sourceWindowBoundary || target[0] > m.targetWindowBoundary {
								if m.matchesInCurrentWindow < config.minimumMatchingNgramsInWindow {
									m.inAlignment = false
								} else {
									if source[0] > m.maxSourceGap || target[0] > m.maxTargetGap {
										m.inAlignment = false
									} else {
										m.sourceAnchor = source[0]
										m.sourceWindowBoundary = m.sourceAnchor + config.matchingWindowSize
										m.targetAnchor = target[0]
										m.targetWindowBoundary = m.targetAnchor + config.matchingWindowSize
										m.matchesInCurrentWindow = 0
									}
								}
							}
							if !m.inAlignment {
								if float32(m.commonNgramMatches/m.matchesInCurrentAlignment) < config.commonNgramsLimit {
									if m.matchesInCurrentAlignment >= config.minimumMatchingNgramsInWindow {
										m.currentAlignment.source = position{m.firstMatch[0][1], m.lastMatch[0][2], m.firstMatch[0][0], m.lastMatch[0][0]}
										m.currentAlignment.target = position{m.firstMatch[1][1], m.lastMatch[1][2], m.firstMatch[1][0], m.lastMatch[1][0]}
										alignments = append(alignments, m.currentAlignment)
										// fmt.Println(m.currentAlignment.source.startIndex, m.currentAlignment.source.endIndex)
										// Looking for small match within max_gap
									} else if (m.lastMatch[0][0]-currentAnchor.source[0]) <= config.maxGap && m.matchesInCurrentAlignment >= config.minimumMatchingNgrams {
										m.currentAlignment.source = position{m.firstMatch[0][1], m.lastMatch[0][2], m.firstMatch[0][0], m.lastMatch[0][0]}
										m.currentAlignment.target = position{m.firstMatch[1][1], m.lastMatch[1][2], m.firstMatch[1][0], m.lastMatch[1][0]}
										alignments = append(alignments, m.currentAlignment)
										// fmt.Println(m.currentAlignment.source.startIndex, m.currentAlignment.source.endIndex)
									}
								}
								m.lastSourcePosition = m.lastMatch[0][0] + 1 // Make sure we start the next match at index that follows last source match
								// if m.currentAlignment.source.endByte != 0 {
								// 	fmt.Println("Failed", source[0], target[0], m.currentAlignment.source.endIndex, m.currentAlignment.target.endIndex)
								// }
								break innerMatchingLoop
							}
							m.lastSourcePosition = source[0]
							m.maxSourceGap = m.lastSourcePosition + config.maxGap
							m.lastTargetPosition = target[0]
							m.maxTargetGap = m.lastTargetPosition + config.maxGap
							m.previousSourceIndex = source[0]
							m.matchesInCurrentWindow++
							m.matchesInCurrentAlignment++
							m.lastMatch = []indexedNgram{source, target} // save last matching ngrams
							if _, ok := mostCommonNgrams[match.ngram]; ok {
								m.commonNgramMatches++
							}
						}
					}
					// Add current alignment if not already done
					if m.inAlignment && m.matchesInCurrentAlignment >= config.minimumMatchingNgrams {
						m.currentAlignment.source = position{m.firstMatch[0][1], m.lastMatch[0][2], m.firstMatch[0][0], m.lastMatch[0][0]}
						m.currentAlignment.target = position{m.firstMatch[1][1], m.lastMatch[1][2], m.firstMatch[1][0], m.lastMatch[1][0]}
						alignments = append(alignments, m.currentAlignment)
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
		for i := 0; i < config.numThreads; i++ {
			localCombinedAlignments := <-c
			if len(localCombinedAlignments) > 0 {
				combinedAlignments.alignments = append(combinedAlignments.alignments, localCombinedAlignments...)
			}
		}
		if len(combinedAlignments.alignments) > 0 {
			writeAligments(&combinedAlignments, &sourceFile.DocID, sourceMetadata, targetMetadata, mergedOutput, &config)
		}
		if sourceAgainstSource {
			sourceFileDone[sourceFile.DocID] = true
		}
	}
	mergedOutput.Sync()
	mergedOutput.Close()
	fmt.Printf("%d results...\n", counts)
}
