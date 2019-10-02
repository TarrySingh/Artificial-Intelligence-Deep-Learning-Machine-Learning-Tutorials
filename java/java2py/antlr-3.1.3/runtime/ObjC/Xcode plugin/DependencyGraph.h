// [The "BSD licence"]
// Copyright (c) 2006 Kay Roepke
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#import "BuildSystemInterfaces.h"

typedef struct {
    unsigned long long hi;
    unsigned long long lo;
} CDAnonymousStruct1;

@interface XCDependencyNode : NSObject
{
    unsigned int _nodeNumber;
    NSString *_name;
    NSString *_path;
    PBXTargetBuildContext *_buildContext;
    NSMutableArray *_producerCommands;
    NSMutableArray *_includedNodes;
    NSMutableArray *_consumerCommands;
    NSMutableArray *_includingNodes;
    struct {
        unsigned int alwaysOutOfDate:1;
        unsigned int dontCareIfExists:1;
        unsigned int dontCareAboutTimestamp:1;
        unsigned int shouldScanForIncludes:1;
        unsigned int beingEnqueued:1;
        unsigned int beingReset:1;
    } _dnFlags;
    NSData *_automaticFileContents;
    id _fileInfoEntityIdent;
    int _state;
    int _time;
    CDAnonymousStruct1 _signature;
    int _highestInclState;
    int _newestInclTime;
    CDAnonymousStruct1 _combinedInclSignature;
    unsigned int _traversalGenCount;
    int _fileSize;
}

- (id)initWithNodeNumber:(unsigned int)fp8 name:(id)fp12 path:(id)fp16;
- (id)initWithNodeNumber:(unsigned int)fp8 name:(id)fp12;
- (id)init;
- (void)dealloc;
- (void)detachFromOtherGraphObjects;
- (id)buildContext;
- (void)setBuildContext:(id)fp8;
- (unsigned int)nodeNumber;
- (id)name;
- (BOOL)isVirtual;
- (id)path;
- (id)paths;
- (id)dependencyInfoCacheEntry;
- (id)producerCommands;
- (id)includedNodes;
- (id)consumerCommands;
- (id)includingNodes;
- (id)producerCommand;
- (id)command;
- (id)automaticFileContents;
- (void)setAutomaticFileContents:(id)fp8;
- (void)setFileInfoEntityIdentifier:(id)fp8;
- (void)incrementWaitCount;
- (void)decrementWaitCount;
- (void)makeConsumerCommandsPerformSelector:(SEL)fp8 withObject:(id)fp12 recursionAvoidanceBitmap:(char *)fp16;
- (void)makeConsumerCommandsPerformSelector:(SEL)fp8 withObject:(id)fp12;
- (void)invalidateSignature;
- (void)invalidateCombinedIncludesSignature;
- (void)_addDependingNode:(id)fp8;
- (void)_addIncludingNode:(id)fp8;
- (void)_removeDependingNode:(id)fp8;
- (void)_removeIncludingNode:(id)fp8;
- (void)addDependedNode:(id)fp8;
- (void)addIncludedNode:(id)fp8;
- (void)removeAllIncludedNodes;
- (void)setScansFileContentsForIncludes:(BOOL)fp8;
- (void)_addProducerCommand:(id)fp8;
- (void)_addConsumerCommand:(id)fp8;
- (BOOL)isAlwaysOutOfDate;
- (void)setAlwaysOutOfDate:(BOOL)fp8;
- (BOOL)dontCareIfExists;
- (void)setDontCareIfExists:(BOOL)fp8;
- (BOOL)dontCareAboutTimestamp;
- (void)setDontCareAboutTimestamp:(BOOL)fp8;
- (BOOL)recordsUpdatedFileListInEnvironment;
- (void)setRecordsUpdatedFileListInEnvironment:(BOOL)fp8;
- (int)state;
- (int)highestStateOfIncludedNodes;
- (long)timestamp;
- (long)newestTimestampOfIncludedNodes;
- (long)fileSize;
- (CDAnonymousStruct1)signature;
- (CDAnonymousStruct1)combinedSignatureOfIncludedNodes;
- (void)setCommandInputSignature:(CDAnonymousStruct1)fp8 commandConfigurationSignature:(CDAnonymousStruct1)fp24;
- (void)statFileIfNeeded;
- (void)computeStateIfNeeded;
- (void)enqueueOutOfDateCommandsOntoWorkQueue:(id)fp8;
- (BOOL)isUpToDate;
- (BOOL)isUpToDateWithRespectToNode:(id)fp8;
- (void)resetState;
- (void)resetStateRecursively;
- (void)removePredictiveProcessingOutputRecursivelyBecauseOfChangedNode:(id)fp8;
- (void)fileMayHaveChanged;
- (void)touch;
- (void)untouch;
- (void)setFileInfo:(id)fp8 forKey:(id)fp12;
- (void)updateDiscoveredBuildInfo;
- (void)removeDiscoveredInfo;
- (id)stateDescription;
- (id)timeDescription;
- (id)signatureDescription;
- (id)shortNameForDebugging;
- (id)description;
- (id)nameForGraphViz;
- (void)writeDefinitionToGraphVizFile:(struct __sFILE *)fp8;
- (void)writeDependencyEdgesToGraphVizFile:(struct __sFILE *)fp8;
- (void)writeInclusionEdgesToGraphVizFile:(struct __sFILE *)fp8;

@end

@class XCWorkQueue;

@protocol XCWorkQueueCommands <NSObject>
- (void)wasAddedToWorkQueue:(id)fp8;
- (void)willBeRemovedFromWorkQueue:(id)fp8;
- (id)workQueue;
- (BOOL)isReadyForProcessing;
- (unsigned int)phaseNumber;
- (id)ruleInfo;
- (void)willActivateInWorkQueue:(id)fp8;
- (void)didDeactivateInWorkQueue:(id)fp8 didCompleteSuccessfully:(BOOL)fp12;
- (unsigned int)waitCount;
- (void)incrementWaitCount;
- (void)decrementWaitCount;
- (void)incrementWaitCountsOfDependingNodes;
- (void)decrementWaitCountsOfDependingNodes;
- (id)createStartedCommandInvocationInSlot:(unsigned int)fp8 ofWorkQueueOperation:(id)fp12;
- (void)commandInvocationWillStart:(id)fp8;
- (void)commandInvocationDidEnd:(id)fp8 successfully:(BOOL)fp12;
- (unsigned int)workQueueCommandTag;
- (void)setWorkQueueCommandTag:(unsigned int)fp8;
- (id)predictiveProcessingCandidateFilePath;
- (id)predictiveProcessingValiditySignature;
- (id)subprocessCommandLineForProcessing;
- (id)descriptionForWorkQueueLog;
- (id)instantiatedCommandOutputParserWithDelegate:(id)fp8;
@end

@interface XCWorkQueueCommand : NSObject <XCWorkQueueCommands>
{
    XCWorkQueue *_workQueue;
    unsigned int _workQueueCommandTag;
    unsigned int _waitCount;
}

- (id)init;
- (void)dealloc;
- (void)finalize;
- (id)workQueue;
- (void)wasAddedToWorkQueue:(id)fp8;
- (void)willBeRemovedFromWorkQueue:(id)fp8;
- (unsigned int)workQueueCommandTag;
- (void)setWorkQueueCommandTag:(unsigned int)fp8;
- (unsigned int)phaseNumber;
- (id)ruleInfo;
- (BOOL)isReadyForProcessing;
- (unsigned int)waitCount;
- (void)incrementWaitCount;
- (void)decrementWaitCount;
- (void)incrementWaitCountsOfDependingNodes;
- (void)decrementWaitCountsOfDependingNodes;
- (void)willActivateInWorkQueue:(id)fp8;
- (id)createStartedCommandInvocationInSlot:(unsigned int)fp8 ofWorkQueueOperation:(id)fp12;
- (id)predictiveProcessingCandidateFilePath;
- (id)predictiveProcessingValiditySignature;
- (void)commandInvocationWillStart:(id)fp8;
- (void)commandInvocationDidEnd:(id)fp8 successfully:(BOOL)fp12;
- (void)didDeactivateInWorkQueue:(id)fp8 didCompleteSuccessfully:(BOOL)fp12;
- (id)subprocessCommandLineForProcessing;
- (id)descriptionForWorkQueueLog;
- (id)instantiatedCommandOutputParserWithDelegate:(id)fp8;

@end

@class XCWorkQueueCommandInvocation, XCPropertyDictionary;
@interface XCDependencyCommand : XCWorkQueueCommand
{
    PBXTargetBuildContext *_buildContext;
    unsigned int _commandNumber;
    BOOL _waitCountBeingAdjusted;
    BOOL _beingEnqueued;
    BOOL _caresAboutIncludes;
    BOOL _mightHavePredProcOutput;
    NSMutableArray *_filePathsToRemove;
    unsigned int _phaseNumber;
    NSMutableArray *_inputNodes;
    NSMutableArray *_outputNodes;
    XCCommandLineToolSpecification *_toolSpecification;
    NSArray *_ruleInfo;
    NSString *_commandPath;
    NSMutableArray *_arguments;
    NSMutableDictionary *_environment;
    NSString *_workingDirPath;
    id _customToolInfoObject;
    CDAnonymousStruct1 _inputSignature;
    CDAnonymousStruct1 _configSignature;
    int _state;
    NSString *_whyState;
    XCWorkQueueCommandInvocation *_currentInvocation;
    NSMutableString *_commandLineDisplayString;
    XCPropertyDictionary *_launchPropertyDict;
}

- (id)initWithCommandNumber:(unsigned int)fp8 ruleInfo:(id)fp12 commandPath:(id)fp16 arguments:(id)fp20 environment:(id)fp24;
- (id)initWithCommandNumber:(unsigned int)fp8 ruleInfo:(id)fp12 commandPath:(id)fp16 arguments:(id)fp20;
- (id)initWithCommandNumber:(unsigned int)fp8 ruleInfo:(id)fp12 commandPath:(id)fp16;
- (id)init;
- (void)dealloc;
- (void)detachFromOtherGraphObjects;
- (id)buildContext;
- (void)setBuildContext:(id)fp8 commandNumber:(unsigned int)fp12;
- (id)name;
- (unsigned int)phaseNumber;
- (void)setPhaseNumber:(unsigned int)fp8;
- (void)invalidateInputSignature;
- (void)invalidateConfigurationSignature;
- (id)inputNodes;
- (void)addInputNode:(id)fp8;
- (id)outputNodes;
- (void)addOutputNode:(id)fp8;
- (void)_addDependingNode:(id)fp8;
- (id)toolSpecification;
- (void)setToolSpecification:(id)fp8;
- (unsigned int)commandNumber;
- (id)ruleInfo;
- (void)setRuleInfo:(id)fp8;
- (id)commandPath;
- (void)setCommandPath:(id)fp8;
- (id)arguments;
- (void)addArgument:(id)fp8;
- (void)addArguments:(id)fp8;
- (void)addArgumentsFromArray:(id)fp8;
- (unsigned int)numberOfArguments;
- (id)argumentAtIndex:(unsigned int)fp8;
- (unsigned int)indexOfArgumentHavingPrefix:(id)fp8 startingAtIndex:(unsigned int)fp12;
- (void)replaceArgumentAtIndex:(unsigned int)fp8 withArgument:(id)fp12;
- (unsigned int)transformArgumentsHavingPrefix:(id)fp8 inRange:(struct _NSRange)fp12 usingFormatString:(id)fp20;
- (unsigned int)transformArgumentsHavingPrefix:(id)fp8 usingFormatString:(id)fp12;
- (id)commandLine;
- (id)environment;
- (void)setEnvironment:(id)fp8;
- (void)addEnvironmentValue:(id)fp8 forKey:(id)fp12;
- (void)addEnvironmentEntriesFromDictionary:(id)fp8;
- (id)workingDirectoryPath;
- (void)setWorkingDirectoryPath:(id)fp8;
- (BOOL)caresAboutIncludes;
- (void)setCaresAboutIncludes:(BOOL)fp8;
- (id)filePathsToRemove;
- (void)addFilePathToRemove:(id)fp8;
- (void)_addConfigurationSignatureIngredientsToMD5Context:(struct CC_MD5state_st *)fp8;
- (CDAnonymousStruct1)inputSignature;
- (CDAnonymousStruct1)configurationSignature;
- (id)predictiveProcessingCandidateFilePath;
- (id)predictiveProcessingValiditySignature;
- (void)noteMightHavePredictiveProcessingOutput;
- (BOOL)isReadyForProcessing;
- (void)checkWaitCounts;
- (id)createStartedCommandInvocationInSlot:(unsigned int)fp8 ofWorkQueueOperation:(id)fp12;
- (void)commandInvocationWillStart:(id)fp8;
- (void)commandInvocationDidEnd:(id)fp8 successfully:(BOOL)fp12;
- (id)subprocessCommandLineForProcessing;
- (id)subprocessWorkingDirectoryForProcessing;
- (id)subprocessExtraEnvironmentEntriesForProcessing;
- (id)descriptionForWorkQueueLog;
- (id)instantiatedCommandOutputParserWithDelegate:(id)fp8;
- (void)incrementWaitCount;
- (void)decrementWaitCount;
- (void)incrementWaitCountsOfDependingNodes;
- (void)decrementWaitCountsOfDependingNodes;
- (id)displayString;
- (id)shortNameForDebugging;
- (id)stateDescription;
- (id)signatureDescription;
- (id)description;
- (id)nameForGraphViz;
- (void)writeDefinitionToGraphVizFile:(struct __sFILE *)fp8;
- (void)writeInputEdgesToGraphVizFile:(struct __sFILE *)fp8;
- (int)state;
- (CDAnonymousStruct1)signature;
- (id)path;
- (id)paths;
- (void)statFileIfNeeded;
- (void)resetState;
- (void)resetStateRecursively;
- (BOOL)isUpToDate;
- (void)computeStateIfNeeded;
- (BOOL)needsToRun;
- (void)enqueueOutOfDateCommandsOntoWorkQueue:(id)fp8;
- (void)willActivateInWorkQueue:(id)fp8;
- (void)propagateSignatureToOutputNodes;
- (void)didDeactivateInWorkQueue:(id)fp8 didCompleteSuccessfully:(BOOL)fp12;
- (id)launchPropertyExpansionDictionary;
- (void)setLaunchPropertyExpansionDictionary:(id)fp8;
- (id)customToolInfoObject;
- (void)setCustomToolInfoObject:(id)fp8;
- (void)makeOutputNodesPerformSelector:(SEL)fp8 withObject:(id)fp12;
- (void)makeConsumerCommandsOfOutputNodesPerformSelector:(SEL)fp8 withObject:(id)fp12;
- (void)removePredictiveProcessingOutputRecursivelyBecauseOfChangedNode:(id)fp8;
- (void)fileMayHaveChanged;
- (void)touch;
- (void)untouch;
- (id)dependencyNode;

@end
