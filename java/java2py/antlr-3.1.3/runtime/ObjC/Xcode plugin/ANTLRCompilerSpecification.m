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

#import "ANTLRCompilerSpecification.h"
#import "DependencyGraph.h"
#import "BuildSystemInterfaces.h"
#import "GrammarFilterLexer.h"
#import <ANTLR/ANTLR.h>

@implementation ANTLRCompilerSpecification

- (NSArray *)computeDependenciesForFilePath:(NSString *)inputPath ofType:(PBXFileType *)type outputDirectory:(NSString *)output inTargetBuildContext:(PBXTargetBuildContext *)context
{
	NSMutableArray *result = [NSMutableArray arrayWithCapacity:6];
	[self setGrammarName:nil];
	[self setGrammarType:nil];
	[self setDependsOnVocab:nil];
	[self setIsCombinedGrammar:NO];
	
	// expand the input file
	inputPath = [context expandedValueForString:inputPath];
	XCDependencyNode *inputNode = [context dependencyNodeForName:inputPath createIfNeeded:YES];
	[inputNode setScansFileContentsForIncludes:YES];
	//[context writeToGraphVizFileAtPath:@"/tmp/antlr_xcode_before.dot"];
	
	// create the lexer to check dependencies and grammar type
	NSError *error;
	NSString *grammarContent = [NSString stringWithContentsOfFile:inputPath encoding:NSASCIIStringEncoding error:&error];
	if (grammarContent == nil) { NSLog(@"couldn't read grammar: %@", error); }
	ANTLRStringStream *inputStream = [[ANTLRStringStream alloc] initWithString:grammarContent];
	GrammarFilterLexer *grammarLexer = [[GrammarFilterLexer alloc] initWithCharStream:inputStream];
	// make it report its findings to us
	[grammarLexer setDelegate:self];
	ANTLRToken *currentToken;
	while (currentToken = [grammarLexer nextToken]) ;	// chew through the entire file to look for aliens
	[grammarLexer release];
	[inputStream release];
	
	// determine the output file paths
	// this is a bit tricky, because we actually have more than one file, and their names depend on the contents of the grammar...
	// this way it only depends on the type of grammar:
	//		`grammar'	=>	`grammar'Parser.(m|h)
	//		`combined'	=>  `grammar'Parser.(m|h) and `grammar'Lexer.(m|h) output files
	//		`lexer'		=>	`grammar'Lexer.(m|h)
	//		`tree'		=>	`grammar'TreeParser.(m|h)
	// DONE: actually look inside the file to get the actual grammar name & type -> use ANTLR filtering lexer for this
	// TODO: at least support java grammars, too, not only ObjC
	// DONE: maybe move the whole shebang to $(DERIVED_FILE_DIR)?
	// DONE: also create nodes for .tokens files if generated
	// DONE: *must* analyze all grammars for dependencies (tokenVocab!) -> solved by creating dep nodes (includes) for .tokens files
	// TODO: figure out how to support "clean" build phase
	NSString *grammarNameFromFileName = [[inputPath lastPathComponent] stringByDeletingPathExtension];		// somepath/grammar.g -> grammar
	NSString *outputDirectory = [context expandedValueForString:@"$(ANTLR_FORCE_OUT_DIR)"];
	[context createDirectoryAtPath:outputDirectory];

	NSMutableArray *outputFileSuffixes = [[NSMutableArray alloc] initWithCapacity:2];
	// figure out which grammar types will be generated and put those in the queue
	if ([grammarType isEqualToString:@"parser"]) {
		[outputFileSuffixes addObject:@"Parser"];
		if (isCombinedGrammar) {
			[outputFileSuffixes addObject:@"Lexer"];
			[self setGrammarType:@"combined"];
		}
	} else if ([grammarType isEqualToString:@"lexer"]) {
		[outputFileSuffixes addObject:@"Lexer"];
	} else if ([grammarType isEqualToString:@"tree"]) {
		[outputFileSuffixes addObject:@"TreeParser"];
	}
	
	// if this grammar imports a token vocab, get the node for that and add it as an include node to the input file
	XCDependencyNode *importedTokensNode = nil;
	if (importedVocabName) {
		NSString *tokensFile = [outputDirectory stringByAppendingPathComponent:[importedVocabName stringByAppendingPathExtension:@"tokens"]];
		importedTokensNode = [context dependencyNodeForName:[context expandedValueForString:tokensFile] createIfNeeded:YES];
		[inputNode addIncludedNode:importedTokensNode];	// is this the correct order?
	}
	
	//NSLog(@"found grammar %@ in file %@: type:%@ imports tokenVocab: %@", grammarName, inputPath, grammarType, importedVocabName);

	// create the command to execute ANTLR and set the necessary arguments
	XCDependencyCommand *depCmd = [context
		createCommandWithRuleInfo:[NSArray arrayWithObjects:@"ANTLR", [context naturalPathForPath:inputPath], nil]
					  commandPath:[[context expandedValueForString:[self path]] stringByStandardizingPath]
						arguments:nil
						  forNode:nil];		// the output nodes are added below!
	[depCmd setToolSpecification:self];
	[depCmd addArgument:@"-cp"];
	if ([[context expandedValueForString:@"$(ANTLR_EXTRA_JAVA_ARGS)"] isEqualToString:@""]) {
		[context addDependencyAnalysisErrorMessageFormat:@"You must supply the Java classpath to ANTLR (both v2 and v3) & StringTemplate! (ANTLR_EXTRA_JAVA_ARGS build setting)"];
		return nil;
	}
	[depCmd addArgument:[context expandedValueForString:@"$(ANTLR_EXTRA_JAVA_ARGS)"]];
	[depCmd addArgument:@"org.antlr.Tool"];
	[depCmd addArgument:@"-message-format"];
	[depCmd addArgument:@"gnu"];
	[depCmd addArgumentsFromArray:[self commandLineForAutogeneratedOptionsInTargetBuildContext:context]];
/*	if ([context expandedValueIsNonEmptyForString:@"$(ANTLR_LIB_DIR)"]) {
		[depCmd addArgument:@"-l"];
		[depCmd addArgument:[context expandedValueForString:@"$(ANTLR_LIB_DIR)"]];
	}
*/
	[depCmd addArgument:inputPath];

	
	NSEnumerator *suffixEnumerator = [outputFileSuffixes objectEnumerator];
	NSString *suffix = nil;
	while (suffix = (NSString *)[suffixEnumerator nextObject]) {
		// output file names
		NSString *baseName = [outputDirectory stringByAppendingPathComponent:[grammarNameFromFileName stringByAppendingString:suffix]];
		NSString *sourceFile = [context expandedValueForString:[baseName stringByAppendingPathExtension:@"m"]];
		NSString *headerFile = [context expandedValueForString:[baseName stringByAppendingPathExtension:@"h"]];
		NSString *tokensFile = [context expandedValueForString:[outputDirectory stringByAppendingPathComponent:[grammarNameFromFileName stringByAppendingPathExtension:@"tokens"]]];
		XCDependencyNode *sourceFileNode = [context dependencyNodeForName:sourceFile createIfNeeded:YES];
		XCDependencyNode *headerFileNode = [context dependencyNodeForName:headerFile createIfNeeded:YES];
		XCDependencyNode *tokensFileNode = [context dependencyNodeForName:tokensFile createIfNeeded:YES];
		
		// create dependency graph egdes
		[depCmd addOutputNode:sourceFileNode];
		[depCmd addOutputNode:headerFileNode];
		[depCmd addOutputNode:tokensFileNode];
		[sourceFileNode addDependedNode:inputNode];
		[headerFileNode addDependedNode:inputNode];
		[tokensFileNode addDependedNode:inputNode];
		
		// setup mapping grammar file <-> generated code
		[context setCompiledFilePath:sourceFile forSourceFilePath:inputPath];
		[context setCompiledFilePath:headerFile forSourceFilePath:inputPath];
		[context setCompiledFilePath:tokensFile forSourceFilePath:inputPath];
		
		// tell Xcode which files our compiler will create, so Xcode can send that output on to other commands further down the chain 
		[result addObject:sourceFile];
		[result addObject:headerFile];
		[result addObject:tokensFile];
	}
	[outputFileSuffixes release];
	
	//[context writeToGraphVizFileAtPath:@"/tmp/antlr_xcode.dot"];
	return result;
}

// delegate methods of GrammarFilterLexer

- (void) setGrammarName:(NSString *)theGrammarName
{
	if (grammarName != theGrammarName) {
		[grammarName release];
		grammarName = [theGrammarName retain];
	}
}

- (void) setGrammarType:(NSString *)theGrammarType
{
	if (grammarType != theGrammarType) {
		[grammarType release];
		grammarType = [theGrammarType retain];
	}
}

- (void) setIsCombinedGrammar:(BOOL)combinedGrammar
{
	isCombinedGrammar = combinedGrammar;
}

- (void) setDependsOnVocab:(NSString *)theVocabName
{
	if (importedVocabName != theVocabName) {
		[importedVocabName release];
		importedVocabName = [theVocabName retain];
	}
}

@end
