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

@class XCBuildOperation, XCPropertyExpansionContext;

@interface PBXBuildContext : NSObject
{
    XCBuildOperation *_currentBuildOperation;
    NSString *_baseDirectoryPath;
    XCPropertyExpansionContext *_propertyExpansionContext;
}

- (id)init;
- (void)dealloc;
- (id)currentBuildOperation;
- (void)setCurrentBuildOperation:(id)fp8;
- (id)baseDirectoryPath;
- (void)setBaseDirectoryPath:(id)fp8;
- (void)setStringValue:(id)fp8 forDynamicSetting:(id)fp12;
- (void)removeDynamicSetting:(id)fp8;
- (void)prependStringOrStringListValue:(id)fp8 toDynamicSetting:(id)fp12;
- (void)appendStringOrStringListValue:(id)fp8 toDynamicSetting:(id)fp12;
- (void)removeStringOrStringListValue:(id)fp8 fromDynamicSetting:(id)fp12;
- (void)removeAllDynamicSettings;
- (void)pushToolParameterTierBuildSettingsDictionary:(id)fp8;
- (void)popToolParameterTierBuildSettingsDictionary;
- (void)setToolParameterTierBuildSettingsDictionaries:(id)fp8;
- (void)setHighestTierBuildSettingsDictionaries:(id)fp8;
- (void)pushCustomTierBuildSettingsDictionary:(id)fp8;
- (void)popCustomTierBuildSettingsDictionary;
- (void)setCustomTierBuildSettingsDictionaries:(id)fp8;
- (void)pushDefaultsTierBuildSettingsDictionary:(id)fp8;
- (void)popDefaultsTierBuildSettingsDictionary;
- (void)setDefaultsTierBuildSettingsDictionaries:(id)fp8;
- (void)setLowestTierBuildSettingsDictionaries:(id)fp8;
- (id)propertyExpansionContext;
- (void)setPropertyExpansionContext:(id)fp8;
- (id)expandedValueForString:(id)fp8;
- (BOOL)expandedValueIsNonEmptyForString:(id)fp8;
- (BOOL)expandedBooleanValueForString:(id)fp8;
- (id)arrayByExpandingEntriesInArray:(id)fp8;
- (id)dictionaryByExpandingEntriesInDictionary:(id)fp8;
- (id)absoluteExpandedPathForString:(id)fp8;

@end

@class NSArray, NSCountedSet, NSDictionary, NSMutableArray, NSMutableDictionary, NSMutableSet, NSRecursiveLock, NSString, NSThread, PBXTarget, XCBuildInfoManager, XCDependencyInfoCache, XCHierarchicalOutputLog, XCTargetDGSnapshot;

@interface PBXTargetBuildContext : PBXBuildContext
{
    PBXTarget *_target;
    NSString *_presumedBuildAction;
    NSString *_presumedConfigName;
    NSMutableArray *_depGraphEvents;
    NSRecursiveLock *_depGraphLock;
    unsigned int _depGraphLockLevel;
    NSThread *_depGraphAccessorThread;
    BOOL _needsDependencyGraphCreation;
    BOOL _shouldCancelDependencyGraphCreation;
    BOOL _isCreatingDependencies;
    BOOL _hasSetUpBuildSettings;
    NSString *_productDirectoryPath;
    NSString *_buildDirectoryPath;
    NSMutableDictionary *_cachedHeadermaps;
    NSMutableDictionary *_headerSearchContexts;
    NSMutableDictionary *_indexingInfoDicts;
    NSMutableDictionary *_effectiveSearchPaths;
    NSMutableArray *_searchPathsForRez;
    NSMutableArray *_nodes;
    NSMutableArray *_commands;
    NSMutableDictionary *_nodesByName;
    NSMutableArray *_depAnalysisMessageStrings;
    NSMutableArray *_buildSetupMessageStrings;
    unsigned int _numDepAnalysisErrors;
    unsigned int _numDepAnalysisWarnings;
    NSMutableDictionary *_auxiliaryFilesData;
    NSMutableDictionary *_auxiliaryFilesPermissions;
    NSMutableDictionary *_auxiliarySymlinksContents;
    NSMutableDictionary *_filePathsToBuildFileRefs;
    NSMutableDictionary *_sourcesToObjFiles;
    NSMutableDictionary *_objFilesToSources;
    NSCountedSet *_countedBuildFileBaseNames;
    NSMutableDictionary *_constantBuildToolFlags;
    NSMutableDictionary *_filePathLists;
    NSArray *_pathPrefixesExcludedFromHeaderDependencies;
    NSMutableSet *_activeToolSpecs;
    unsigned int _currentPhaseNumber;
    BOOL _autoIncrementPhaseNumber;
    NSMutableDictionary *_extraLinkerParameters;
    XCHierarchicalOutputLog *_debugOutputLog;
    unsigned int _debugOutputEnableCount;
    BOOL _enableDistBuilds;
    NSArray *_distBuildsServerList;
    NSDictionary *_distBuildsEnvEntries;
    unsigned int _distBuildsParallelTasks;
    NSMutableArray *_linkerBuildMessages;
    NSMutableArray *_otherBuildMessages;
    XCBuildInfoManager *_buildInfoManager;
    XCDependencyInfoCache *_dependencyInfoCache;
    NSMutableArray *_productNodes;
    XCTargetDGSnapshot *_targetSnapshotForDG;
    NSMutableArray *_nodesThatNeedToSetBuildInfo;
    NSMutableSet *_derivedFileCaches;
}

+ (void)initialize;
+ (id)identifierForHeadermapWithBreadthFirstRecursiveContentsAtPath:(id)fp8;
+ (id)identifierForHeadermapWithGeneratedFiles;
+ (id)identifierForHeadermapWithProductHeaders;
+ (id)identifierForHeadermapWithAllProductHeadersInProject;
+ (id)identifierForHeadermapWithAllHeadersInProject;
+ (id)identifierForTraditionalHeadermap;
+ (id)headerFileExtensionsForHeadermaps;
- (id)initWithTarget:(id)fp8;
- (void)dealloc;
- (void)finalize;
- (id)target;
- (void)targetWillDealloc:(id)fp8;
- (id)presumedBuildAction;
- (void)setPresumedBuildAction:(id)fp8;
- (id)presumedBuildConfigurationName;
- (void)setPresumedBuildConfigurationName:(id)fp8;
- (id)targetSnapshot;
- (void)_projectWillClose:(id)fp8;
- (void)_activeBuildConfigurationNameDidChange:(id)fp8;
- (void)disableCacheInvalidation;
- (void)enableCacheInvalidation;
- (BOOL)shouldUseDistributedBuilds;
- (id)baseDirectoryPath;
- (id)absolutePathForPath:(id)fp8;
- (id)naturalPathForPath:(id)fp8;
- (BOOL)lockDependencyGraphBeforeDate:(id)fp8;
- (void)lockDependencyGraph;
- (void)unlockDependencyGraph;
- (BOOL)doesCurrentThreadHoldDependencyGraphLock;
- (void)createDependencyGraphWithTargetDGSnapshot:(id)fp8;
- (void)removeSearchPathArgumentsFromArrayOfCommandLineArguments:(id)fp8;
- (id)_searchPathsForCurrentStateAndOtherFlagsBuildSettingsName:(id)fp8 builtinSystemHeaderSearchPaths:(id)fp12 builtinFrameworkSearchPaths:(id)fp16;
- (id)headerFileSearchContextForSourceFilesUsingCompiler:(id)fp8 languageDialect:(id)fp12;
- (id)preprocessingInfoDictionaries;
- (id)preprocessingInfoForIndexingSourceFilesWithCompiler:(id)fp8 languageDialect:(id)fp12;
- (id)dependencyNodeForName:(id)fp8 createIfNeeded:(BOOL)fp12;
- (unsigned int)currentPhaseNumber;
- (void)incrementCurrentPhaseNumber;
- (BOOL)autoIncrementsPhaseNumber;
- (void)setAutoIncrementsPhaseNumber:(BOOL)fp8;
- (id)compilerRequestedLinkerParameters;
- (void)addCompilerRequestedLinkerParameters:(id)fp8;
- (void)removeAllCompilerRequestedLinkerParameters;
- (id)dependencyNodeForName:(id)fp8;
- (unsigned int)numberOfDependencyNodes;
- (id)dependencyNodeWithNumber:(unsigned int)fp8;
- (void)_addMappingFromPath:(id)fp8 toNode:(id)fp12;
- (id)createInvocationOfToolWithIdentifier:(id)fp8 parameterDictionary:(id)fp12;
- (id)createInvocationOfToolWithIdentifier:(id)fp8 parameters:(id)fp12;
- (void)registerDependencyCommand:(id)fp8;
- (void)unregisterDependencyCommand:(id)fp8;
- (id)createCommandWithRuleInfo:(id)fp8 commandPath:(id)fp12 arguments:(id)fp16 forNode:(id)fp20;
- (id)createCommandWithPath:(id)fp8 ruleInfo:(id)fp12;
- (unsigned int)numberOfCommands;
- (id)commandWithNumber:(unsigned int)fp8;
- (BOOL)shouldCancelDependencyGraphCreation;
- (void)cancelDependencyGraphCreation;
- (id)dependencyAnalysisMessageStrings;
- (void)removeAllBuildSetupMessageStrings;
- (unsigned int)numberOfDependencyAnalysisErrors;
- (unsigned int)numberOfDependencyAnalysisWarnings;
- (void)_addDependencyAnalysisMessageString:(id)fp8;
- (void)addDependencyAnalysisErrorMessageFormat:(id)fp8;
- (void)addDependencyAnalysisWarningMessageFormat:(id)fp8;
- (void)defineFileContents:(id)fp8 forAuxiliaryFileAtPath:(id)fp12 withPosixPermissions:(unsigned long)fp16;
- (id)fileContentsForAuxiliaryFileAtPath:(id)fp8;
- (void)defineFileContents:(id)fp8 forAuxiliaryFileAtPath:(id)fp12;
- (id)symlinkContentsForAuxiliarySymlinkAtPath:(id)fp8;
- (void)defineSymlinkContents:(id)fp8 forAuxiliarySymlinkAtPath:(id)fp12;
- (void)setCompiledFilePath:(id)fp8 forSourceFilePath:(id)fp12;
- (void)addActiveToolSpecification:(id)fp8;
- (void)setConstantFlags:(id)fp8 forBuildToolWithIdentifier:(id)fp12;
- (void)addPath:(id)fp8 toFilePathListWithIdentifier:(id)fp12;
- (id)filePathListWithIdentifier:(id)fp8;
- (id)buildFileRefForPath:(id)fp8;
- (id)compiledFilePathForSourceFilePath:(id)fp8;
- (id)sourceFilePathForCompiledFilePath:(id)fp8;
- (void)countBuildFileBaseName:(id)fp8;
- (unsigned int)countForBuildFileBaseName:(id)fp8;
- (id)constantFlagsForBuildToolWithIdentifier:(id)fp8;
- (id)preprocessedFilePathForSourceFilePath:(id)fp8;
- (id)disassembledFilePathForSourceFilePath:(id)fp8;
- (id)activeToolSpecifications;
- (void)pruneDerivedFileCaches;
- (void)addDerivedFileCache:(id)fp8;
- (id)headermapForIdentifier:(id)fp8;
- (void)setHeadermap:(id)fp8 forIdentifier:(id)fp12;
- (id)effectiveSearchPathsForSearchPath:(id)fp8;
- (id)_effectiveSearchPathsForSearchPathBuildSetting:(id)fp8;
- (id)effectiveHeaderSearchPaths;
- (id)effectiveUserHeaderSearchPaths;
- (id)effectiveFrameworkSearchPaths;
- (id)effectiveLibrarySearchPaths;
- (id)effectiveRezSearchPaths;
- (id)searchPathsForRez;
- (BOOL)isFileUpToDateAtPath:(id)fp8;
- (id)buildInfoManager;
- (id)buildInfoValueForKey:(id)fp8 ofEntityIdentifier:(id)fp12;
- (void)setBuildInfoValue:(id)fp8 forKey:(id)fp12 ofEntityIdentifier:(id)fp16;
- (void)removeAllBuildInfoForIdentifier:(id)fp8;
- (id)fileInfoValueForKey:(id)fp8 forFileAtPath:(id)fp12;
- (void)setFileInfoValue:(id)fp8 forKey:(id)fp12 forFileAtPath:(id)fp16;
- (void)clearCompiledFileInfoForFileAtPath:(id)fp8;
- (void)clearCompiledFileInfoForAllFiles;
- (BOOL)areFileInfoNotificationsEnabled;
- (void)disableFileInfoNotifications;
- (void)enableFileInfoNotifications;
- (id)dependencyInfoCacheFilename;
- (id)dependencyInfoCache;
- (id)readDependencyInfoCacheFromBuildDirectory:(id)fp8;
- (id)writeDependencyInfoCacheToBuildDirectory:(id)fp8;
- (id)buildMessagesForFileAtPath:(id)fp8;
- (void)addBuildMessage:(id)fp8 forFileAtPath:(id)fp12;
- (void)removeAllBuildMessagesForFileAtPath:(id)fp8;
- (id)linkerBuildMessages;
- (void)addLinkerBuildMessage:(id)fp8;
- (void)removeAllLinkerBuildMessages;
- (id)uncategorizedBuildMessages;
- (void)addUncategorizedBuildMessage:(id)fp8;
- (void)removeAllUncategorizedBuildMessages;
- (id)productDirectoryPath;
- (id)buildDirectoryPath;
- (id)createDirectoryAtPath:(id)fp8;
- (id)touchFileAtPath:(id)fp8;
- (id)copyFileAtPath:(id)fp8 toPath:(id)fp12;
- (id)dittoFileAtPath:(id)fp8 toPath:(id)fp12;
- (id)moveFileAtPath:(id)fp8 toPath:(id)fp12;
- (id)makeSymlinkToFileAtPath:(id)fp8 atPath:(id)fp12;
- (BOOL)shouldScanHeadersOfFileAtPath:(id)fp8;
- (id)importedFilesForPath:(id)fp8 ensureFilesExist:(BOOL)fp12;
- (id)importedFilesForPath:(id)fp8;
- (BOOL)writeAuxiliaryFilesForBuildOperation:(id)fp8;
- (id)productNodes;
- (void)addProductNode:(id)fp8;
- (void)resetStatesOfAllDependencyNodes;
- (void)analyzeDependenciesForNodes:(id)fp8;
- (void)analyzeDependenciesForFilePaths:(id)fp8;
- (void)analyzeAllProductDependencies;
- (void)checkWaitCountsOfAllDependencyNodes;
- (void)enqueueOutOfDateCommandsOntoWorkQueue:(id)fp8 startingAtNode:(id)fp12;
- (void)enqueueAllOutOfDateCommandsOntoWorkQueue:(id)fp8;
- (void)_addNodeThatNeedsToSetBuildInfo:(id)fp8;
- (void)_makeNodesSetBuildInfoIfNeeded;
- (BOOL)writeToGraphVizFileAtPath:(id)fp8;
- (void)startLoggingDebugOutputIfAppropriate;
- (void)finishLoggingDebugOutput;
- (id)debugOutputLog;
- (id)description;
- (id)nodesMatchingPattern:(id)fp8;

@end
