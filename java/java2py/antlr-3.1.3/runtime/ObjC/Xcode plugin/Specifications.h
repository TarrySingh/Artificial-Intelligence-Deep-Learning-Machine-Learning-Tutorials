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

@interface XCSpecification : NSObject
{
    NSString *_identifier;
    XCSpecification *_superSpecification;
    NSDictionary *_properties;
    NSDictionary *_localizationDictionary;
    NSBundle *_bundle;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (id)_typesToSpecTypeBaseClassesRegistry;
+ (id)_pathExensionsToTypesRegistry;
+ (void)registerSpecificationTypeBaseClass:(Class)fp8;
+ (void)registerSpecificationOrProxy:(id)fp8;
+ (id)registerSpecificationProxyFromPropertyList:(id)fp8;
+ (id)_registerSpecificationProxiesOfType:(id)fp8 fromDictionaryOrArray:(id)fp12 inDirectory:(id)fp16 bundle:(id)fp20 sourceDescription:(id)fp24;
+ (id)registerSpecificationProxiesFromPropertyListsInDirectory:(id)fp8 recursively:(BOOL)fp12 inBundle:(id)fp16;
+ (id)registerSpecificationProxiesFromPropertyListsInDirectory:(id)fp8 recursively:(BOOL)fp12;
+ (id)specificationForIdentifier:(id)fp8;
+ (id)specificationsForIdentifiers:(id)fp8;
+ (id)registeredSpecifications;
+ (id)_subSpecificationsOfSpecification:(id)fp8;
+ (id)registeredBaseSpecifications;
+ (id)allRegisteredSpecifications;
+ (void)loadSpecificationsWithProperty:(id)fp8;
+ (BOOL)_booleanValueForValue:(id)fp8;
- (id)initWithPropertyListDictionary:(id)fp8;
- (id)initAsMissingSpecificationProxyWithIdentifier:(id)fp8 name:(id)fp12 description:(id)fp16;
- (id)init;
- (void)dealloc;
- (void)finalize;
- (id)superSpecification;
- (id)subSpecifications;
- (BOOL)isKindOfSpecification:(id)fp8;
- (BOOL)isAbstract;
- (BOOL)isNotYetLoadedSpecificationProxy;
- (id)loadedSpecification;
- (BOOL)isMissingSpecificationProxy;
- (id)identifier;
- (id)properties;
- (id)localizationDictionary;
- (id)bundle;
- (id)name;
- (id)localizedDescription;
- (int)identifierCompare:(id)fp8;
- (int)nameCompare:(id)fp8;
- (id)_objectForKeyIgnoringInheritance:(id)fp8;
- (id)objectForKey:(id)fp8;
- (id)stringForKey:(id)fp8;
- (id)arrayForKey:(id)fp8;
- (id)dictionaryForKey:(id)fp8;
- (id)dataForKey:(id)fp8;
- (int)integerForKey:(id)fp8;
- (long long)longLongForKey:(id)fp8;
- (float)floatForKey:(id)fp8;
- (double)doubleForKey:(id)fp8;
- (BOOL)boolForKey:(id)fp8;
- (BOOL)boolForKeyFromProxy:(id)fp8;
- (id)arrayOrStringForKey:(id)fp8;
- (id)valueForUndefinedKey:(id)fp8;
- (id)description;

@end

@interface XCPropertyDomainSpecification : XCSpecification
{
    NSDictionary *_buildOptions;
    NSArray *_orderedBuildOptions;
    NSArray *_optionNamesForCommandLine;
    NSArray *_commonBuildOptions;
    NSArray *_buildOptionCategories;
    XCPropertyDomainSpecification *_specForUserInterface;
    NSDictionary *_flattenedBuildOptions;
    NSArray *_flattenedOrderedBuildOptions;
    NSArray *_flattenedCommonBuildOptions;
    NSArray *_flattenedOptionNamesForCommandLine;
    NSArray *_flattenedOptionCategories;
    NSDictionary *_flattenedDefaultValues;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (id)buildOptions;
- (id)buildOptionNamed:(id)fp8;
- (id)orderedBuildOptions;
- (id)commonBuildOptions;
- (id)buildOptionCategories;
- (id)flattenedOptionCategories;
- (id)namesOfBuildOptionsForCommandLine;
- (id)namesOfFlattenedOptionsForCommandLine;
- (id)specificationToShowInUserInterface;
- (id)defaultValuesForAllOptions;
- (id)_expandedValueForCommandLineBuildOptionNamed:(id)fp8 inBuildContext:(id)fp12;
- (id)commandLineForAutogeneratedOptionsForKey:(id)fp8 inTargetBuildContext:(id)fp12;

@end

@interface XCCommandLineToolSpecification : XCPropertyDomainSpecification
{
    Class _commandInvocationClass;
    NSArray *_outputParserClassesOrRules;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (id)unionedDefaultValuesForAllPropertiesForAllTools;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (id)path;
- (Class)commandInvocationClass;
- (id)commandOutputParserClassesOrParseRules;
- (id)hashStringForCommandLineComponents:(id)fp8 inputFilePaths:(id)fp12 inTargetBuildContext:(id)fp16;
- (id)_expandedValueForCommandLineBuildOptionNamed:(id)fp8 inBuildContext:(id)fp12;
- (id)commandLineForAutogeneratedOptionsInTargetBuildContext:(id)fp8;
- (BOOL)areOutputFilesAffectedByCommandLineArgument:(id)fp8;
- (BOOL)areOutputFilesAffectedByEnvironmentVariable:(id)fp8;
- (id)instantiatedCommandOutputParserWithDelegate:(id)fp8;
- (void)_addNameToDefaultValueMappingsToMutableDictionary:(id)fp8;
- (id)createCommandsInBuildContext:(id)fp8;
- (unsigned int)concurrentExecutionCountInTargetBuildContext:(id)fp8;

@end

@interface XCCompilerSpecification : XCCommandLineToolSpecification
{
    NSMutableArray *_inputFileTypes;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (id)displaySpecifications;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (id)inputFileTypes;
- (BOOL)acceptsInputFileType:(id)fp8;
- (BOOL)isAbstract;
- (id)builtinJambaseRuleName;
- (id)uniqueOutputBaseNameForInputFilePath:(id)fp8 inTargetBuildContext:(id)fp12;
- (id)outputFilesForInputFilePath:(id)fp8 inTargetBuildContext:(id)fp12;
- (id)executablePath;
- (id)defaultOutputDirectory;
- (id)effectiveCompilerSpecificationInPropertyExpansionContext:(id)fp8;
- (id)fileTypeForGccLanguageDialect:(id)fp8;
- (id)adjustedFileTypeForInputFileAtPath:(id)fp8 originalFileType:(id)fp12 inTargetBuildContext:(id)fp16;
- (id)computeDependenciesForInputFile:(id)fp8 ofType:(id)fp12 variant:(id)fp16 architecture:(id)fp20 outputDirectory:(id)fp24 inTargetBuildContext:(id)fp28;
- (id)computeDependenciesForFilePath:(id)fp8 ofType:(id)fp12 outputDirectory:(id)fp16 inTargetBuildContext:(id)fp20;

@end

@interface PBXFileType : XCSpecification
{
    NSArray *_extensions;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (void)registerSpecificationOrProxy:(id)fp8;
+ (id)_extensionToFileTypeDictionary;
+ (id)_lowercasedExtensionToFileTypeDictionary;
+ (id)_magicWordToFileTypeDictionary;
+ (id)_fileNamePatternToFileTypeDictionary;
+ (id)_fileTypeDetectorArray;
+ (id)genericFileType;
+ (id)textFileType;
+ (id)genericFolderType;
+ (id)wrapperFolderType;
+ (id)bestFileTypeForRepresentingFileAtPath:(id)fp8 withFileAttributes:(id)fp12 withLessSpecificFileType:(id)fp16 getExtraFileProperties:(id *)fp20;
+ (id)fileTypeForFileName:(id)fp8 posixPermissions:(unsigned int)fp12 hfsTypeCode:(unsigned long)fp16 hfsCreatorCode:(unsigned long)fp20;
+ (id)fileTypeForFileName:(id)fp8;
+ (id)guessFileTypeForGenericFileAtPath:(id)fp8 withFileAttributes:(id)fp12 getExtraFileProperties:(id *)fp16;
+ (id)fileTypeForPath:(id)fp8 getExtraFileProperties:(id *)fp12;
+ (id)fileTypeForPath:(id)fp8;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (void)finalize;
- (id)extensions;
- (id)hfsTypeCodes;
- (BOOL)isBundle;
- (BOOL)isApplication;
- (BOOL)isLibrary;
- (BOOL)isDynamicLibrary;
- (BOOL)isStaticLibrary;
- (BOOL)isFramework;
- (BOOL)isStaticFramework;
- (BOOL)isProjectWrapper;
- (BOOL)isTargetWrapper;
- (BOOL)isExecutable;
- (BOOL)isExecutableWithGUI;
- (BOOL)isPlainFile;
- (BOOL)isTextFile;
- (BOOL)isSourceCode;
- (BOOL)isDocumentation;
- (BOOL)isFolder;
- (BOOL)isNonWrapperFolder;
- (BOOL)isWrapperFolder;
- (BOOL)includeInIndex;
- (BOOL)isTransparent;
- (BOOL)canSetIncludeInIndex;
- (id)languageSpecificationIdentifier;
- (BOOL)isScannedForIncludes;
- (BOOL)requiresHardTabs;
- (id)extraPropertyNames;
- (id)subpathForWrapperPart:(int)fp8 ofPath:(id)fp12 withExtraFileProperties:(id)fp16;
- (id)fileTypePartForIdentifier:(id)fp8;
- (id)_objectForKeyIgnoringInheritance:(id)fp8;
- (id)description;

@end

@protocol XCProductPartOwners
- (id)productPartForIdentifier:(id)fp8;
- (id)subpartsForProductPart:(id)fp8;
@end


@interface XCProductTypeSpecification : XCSpecification <XCProductPartOwners>
{
    NSDictionary *_defaultBuildSettings;
    NSDictionary *_flattenedDefaultBuildSettings;
    NSSet *_allowedBuildPhaseClasses;
    NSArray *_packageTypes;
    NSArray *_productParts;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (id)defaultTargetName;
- (id)defaultBuildSettings;
- (id)allowedBuildPhaseClasses;
- (id)packageTypes;
- (id)defaultPackageType;
- (id)productParts;
- (id)productPartForIdentifier:(id)fp8;
- (id)subpartsForProductPart:(id)fp8;
- (BOOL)hasInfoPlist;
- (id)iconNamePrefix;
- (void)computeDependenciesInTargetBuildContext:(id)fp8;
- (void)initializeTemporaryBuildSettingsInTargetBuildContext:(id)fp8;
- (void)initializeBuildSettingsInTargetBuildContext:(id)fp8;
- (void)computeProductDependenciesInTargetBuildContext:(id)fp8;
- (void)initializeSearchPathBuildSettingsInTargetBuildContext:(id)fp8;
- (id)_prependSDKPackageToPath:(id)fp8 inTargetBuildContext:(id)fp12;
- (void)_alterSearchPaths:(id)fp8 toUseSDKPackageInTargetBuildContext:(id)fp12;
- (void)alterBuildSettingsToUseSDKPackageInTargetBuildContext:(id)fp8;
- (void)defineAuxiliaryFilesInTargetBuildContext:(id)fp8;
- (void)copyAsideProductInTargetBuildContext:(id)fp8;
- (void)generateDSYMFileForLinkedProductInTargetBuildContext:(id)fp8;
- (void)editSymbolsOfLinkedProductInTargetBuildContext:(id)fp8;
- (void)ranlibLinkedProductInTargetBuildContext:(id)fp8;
- (void)separatelyStripSymbolsOfLinkedProductInTargetBuildContext:(id)fp8;
- (void)_computeDependenciesForOwner:(id)fp8 group:(id)fp12 mode:(id)fp16 onFile:(id)fp20 inTargetBuildContext:(id)fp24;
- (void)changePermissionsOnProductInTargetBuildContext:(id)fp8;
- (void)computeSymlinkDependenciesInTargetBuildContext:(id)fp8;
- (id)computeProductTouchActionInTargetBuildContext:(id)fp8;
- (void)compileAuxiliaryFilesForVariant:(id)fp8 architecture:(id)fp12 inTargetBuildContext:(id)fp16;
- (BOOL)shouldStripSymbolsOfLinkedProductInTargetBuildContext:(id)fp8 separately:(char *)fp12;
- (id)linkerSpecificationForObjectFilesInTargetBuildContext:(id)fp8;
- (void)addBaseLinkerFlagsInTargetBuildContext:(id)fp8;
- (void)addWarningLinkerFlagsInTargetBuildContext:(id)fp8;
- (void)addInstallNameLinkerFlagsInTargetBuildContext:(id)fp8;
- (id)createUniversalBinaryFromThinBinaries:(id)fp8 inTargetBuildContext:(id)fp12;

@end

@class PBXLexicalRules;

@interface PBXLanguageSpecification : XCSpecification
{
    NSDictionary *_syntaxColoringRules;
    NSDictionary *_indentationRules;
    NSString *_scannerClassName;
    Class _scannerClass;
    NSString *_lexerClassName;
    Class _lexerClass;
    PBXLexicalRules *_lexRules;
    BOOL _supportsSyntaxAwareIndenting;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (void)finalize;
- (id)syntaxColoringRules;
- (id)indentationRules;
- (BOOL)supportsSyntaxAwareIndenting;
- (id)sourceScanner;
- (id)sourceLexer;
- (id)loadBaseLexicalRules;
- (id)lexicalRules;

@end

@interface PBXPackageTypeSpecification : XCSpecification
{
    NSDictionary *_defaultBuildSettings;
    NSDictionary *_flattenedDefaultBuildSettings;
    NSString *_productReferenceFileTypeIdentifier;
    PBXFileType *_productReferenceFileType;
    NSString *_productReferenceName;
    BOOL _productReferenceIsLaunchable;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (id)wrapperSubpathForWrapperPart:(int)fp8;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (void)finalize;
- (id)defaultBuildSettings;
- (id)productReferenceFileType;
- (id)productReferenceName;
- (BOOL)productReferenceIsLaunchable;

@end

@interface PBXRuntimeSystemSpecification : XCSpecification
{
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (id)nativeRuntimeSystemSpecificaton;
- (id)infoPlistKey;
- (id)specificResourcePath;

@end

@interface PBXBuildSettingsPaneSpecification : XCSpecification
{
    NSString *_settingsDomainPath;
    NSString *_paneClassName;
    NSString *_paneFollows;
    NSArray *_widgets;
}

+ (Class)specificationBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
+ (id)registeredSpecificationsSorted;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (void)finalize;
- (id)name;
- (id)settingsDomainPath;
- (id)paneClassName;
- (Class)paneClass;
- (id)paneFollows;
- (id)widgets;

@end

@interface PBXSCMSpecification : XCSpecification
{
    NSString *_classBaseName;
    BOOL _canAddDirectories;
    BOOL _canRenameFiles;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (void)finalize;
- (id)classBaseName;
- (BOOL)canAddDirectories;
- (BOOL)canRenameFiles;

@end

@interface XCArchitectureSpecification : XCSpecification
{
    unsigned int _byteOrder;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (unsigned int)byteOrder;

@end

@interface XCPlatformSpecification : XCSpecification
{
    NSArray *_architectures;
}

+ (Class)specificationTypeBaseClass;
+ (id)specificationType;
+ (id)localizedSpecificationTypeName;
+ (id)specificationTypePathExtensions;
+ (id)specificationRegistry;
- (id)initWithPropertyListDictionary:(id)fp8;
- (void)dealloc;
- (id)architectures;

@end


