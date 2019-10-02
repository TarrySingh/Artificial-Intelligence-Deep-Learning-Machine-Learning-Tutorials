// $ANTLR 3.0b5 /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g 2006-11-12 20:15:18

#import "GrammarFilterLexer.h"
#pragma mark Cyclic DFAs
@implementation GrammarFilterLexerDFA13
const static int GrammarFilterLexerdfa13_eot[24] =
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     -1};
const static int GrammarFilterLexerdfa13_eof[24] =
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
     -1};
const static unichar GrammarFilterLexerdfa13_min[24] =
    {9,9,0,42,9,0,0,0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0,0};
const static unichar GrammarFilterLexerdfa13_max[24] =
    {125,125,0,47,122,0,0,0,122,122,122,122,122,122,122,122,122,61,122,122,
     122,59,0,0};
const static int GrammarFilterLexerdfa13_accept[24] =
    {-1,-1,5,-1,-1,4,2,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3};
const static int GrammarFilterLexerdfa13_special[24] =
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,
     -1};
const static int GrammarFilterLexerdfa13_transition[] = {};
const static int GrammarFilterLexerdfa13_transition0[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 16, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition1[] = {21, 21, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 22};
const static int GrammarFilterLexerdfa13_transition2[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition3[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 13, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition4[] = {6, -1, -1, -1, 
	-1, 7};
const static int GrammarFilterLexerdfa13_transition5[] = {19, 19, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, 
	20, 20, 20, 20, 20, 20, 20, 20, 20, -1, -1, -1, -1, -1, -1, -1, 20, 20, 
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
	20, 20, 20, 20, 20, 20, -1, -1, -1, -1, 20, -1, 20, 20, 20, 20, 20, 20, 
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
	20, 20};
const static int GrammarFilterLexerdfa13_transition6[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition7[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition8[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 14, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition9[] = {1, 1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 
	5, 5, 5, 5, -1, -1, 2};
const static int GrammarFilterLexerdfa13_transition10[] = {1, 1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 
	5, 5, 5, 5, -1, -1, 2};
const static int GrammarFilterLexerdfa13_transition11[] = {-1};
const static int GrammarFilterLexerdfa13_transition12[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 11, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition13[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 15, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition14[] = {17, 17, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 18, -1, -1, -1, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 
	-1, 5, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5};
const static int GrammarFilterLexerdfa13_transition15[] = {21, 21, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, 
	20, 20, 20, 20, 20, 20, 20, 20, 20, -1, 22, -1, -1, -1, -1, -1, 20, 20, 
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
	20, 20, 20, 20, 20, 20, -1, -1, -1, -1, 20, -1, 20, 20, 20, 20, 20, 20, 
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
	20, 20};
const static int GrammarFilterLexerdfa13_transition16[] = {17, 17, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18};
const static int GrammarFilterLexerdfa13_transition17[] = {5, 5, -1, -1, 
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
	-1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, 5, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, -1, -1, -1, -1, 
	5, -1, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
	5, 5, 5, 5, 5};


- (id) initWithRecognizer:(ANTLRBaseRecognizer *) theRecognizer
{
	if ((self = [super initWithRecognizer:theRecognizer]) != nil) {
		eot = GrammarFilterLexerdfa13_eot;
		eof = GrammarFilterLexerdfa13_eof;
		min = GrammarFilterLexerdfa13_min;
		max = GrammarFilterLexerdfa13_max;
		accept = GrammarFilterLexerdfa13_accept;
		special = GrammarFilterLexerdfa13_special;
		if (!(transition = calloc(24, sizeof(void*)))) {
			[self release];
			return nil;
		}
		transition[0] = GrammarFilterLexerdfa13_transition9;
		transition[1] = GrammarFilterLexerdfa13_transition10;
		transition[2] = GrammarFilterLexerdfa13_transition;
		transition[3] = GrammarFilterLexerdfa13_transition4;
		transition[4] = GrammarFilterLexerdfa13_transition7;
		transition[5] = GrammarFilterLexerdfa13_transition;
		transition[6] = GrammarFilterLexerdfa13_transition;
		transition[7] = GrammarFilterLexerdfa13_transition;
		transition[8] = GrammarFilterLexerdfa13_transition2;
		transition[9] = GrammarFilterLexerdfa13_transition17;
		transition[10] = GrammarFilterLexerdfa13_transition12;
		transition[11] = GrammarFilterLexerdfa13_transition6;
		transition[12] = GrammarFilterLexerdfa13_transition3;
		transition[13] = GrammarFilterLexerdfa13_transition8;
		transition[14] = GrammarFilterLexerdfa13_transition13;
		transition[15] = GrammarFilterLexerdfa13_transition0;
		transition[16] = GrammarFilterLexerdfa13_transition14;
		transition[17] = GrammarFilterLexerdfa13_transition16;
		transition[18] = GrammarFilterLexerdfa13_transition5;
		transition[19] = GrammarFilterLexerdfa13_transition5;
		transition[20] = GrammarFilterLexerdfa13_transition15;
		transition[21] = GrammarFilterLexerdfa13_transition1;
		transition[22] = GrammarFilterLexerdfa13_transition11;
		transition[23] = GrammarFilterLexerdfa13_transition;
	}
	return self;
}

- (int) specialStateTransition:(int) s
{
	int _s = s;
	switch ( s ) {
 				case 0 : 
 				[[recognizer input] rewind];
 				s = -1;
 				if ( ([self evaluateSyntacticPredicate:@selector(synpred3)]) ) {s = 23;}

 				else if ( (YES) ) {s = 5;}

 				if ( s>=0 ) return s;
 				break;
	}
	if ([recognizer isBacktracking]) {
		[recognizer setIsFailed:YES];
		return -1;
	}
	ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:13 state:_s stream:[recognizer input]];
 
	@throw nvae;
}

- (void) dealloc
{
	free(transition);
	[super dealloc];
}

- (NSString *) description
{
	return @"()* loopback of 39:3: ( ( ( WS )? '//' )=> SL_COMMENT | ( ( WS )? '/*' )=> COMMENT | ( ( WS )? 'tokenVocab' )=> ( WS )? 'tokenVocab' ( WS )? '=' ( WS )? tokenVocab= ID ( WS )? ';' | ( WS )? ID ( WS )? '=' ( WS )? ID ( WS )? ';' )*";
}

@end



/** As per Terence: No returns for lexer rules!
#pragma mark Rule return scopes start
#pragma mark Rule return scopes end
*/
@implementation GrammarFilterLexer


- (void) setDelegate:(id)theDelegate
{
	delegate = theDelegate;	// not retained, will always be the object creating this lexer!
}


- (id) initWithCharStream:(id<ANTLRCharStream>)anInput
{
	if (nil!=(self = [super initWithCharStream:anInput])) {



		dfa13 = [[GrammarFilterLexerDFA13 alloc] initWithRecognizer:self];
	}
	return self;
}

- (void) dealloc
{
	[dfa13 release];
	[super dealloc];
}

- (ANTLRToken *) nextToken
{
	[self setToken:nil];
    tokenStartCharIndex = [self charIndex];
    while (YES) {
        if ( [input LA:1] == ANTLRCharStreamEOF ) {
            return nil; // should really be a +eofToken call here -> go figure
        }
        @try {
            int m = [input mark];
            backtracking = 1;
            failed = NO;
            [self mTokens];
            backtracking = 0;
            [input rewind:m];
            if ( failed ) {
                [input consume]; 
            } else {
                [self mTokens];
                return token;
            }
        }
        @catch (ANTLRRecognitionException *re) {
            // shouldn't happen in backtracking mode, but...
            [self reportError:re];
            [self recover:re];
        }
    }
}

- (void) mGRAMMAR
{
	ANTLRToken * _grammarType = nil;
	ANTLRToken * _grammarName = nil;

	@try {
		int _type = GrammarFilterLexer_GRAMMAR;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:25:4: ( (grammarType= GRAMMAR_TYPE WS | ) 'grammar' WS grammarName= ID ( WS )? ';' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:25:4: (grammarType= GRAMMAR_TYPE WS | ) 'grammar' WS grammarName= ID ( WS )? ';' // alt
		{
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:25:4: (grammarType= GRAMMAR_TYPE WS | ) // block
		int alt1=2;
		{
			int LA1_0 = [input LA:1];
			if ( LA1_0=='l'||LA1_0=='p'||LA1_0=='t' ) {
				alt1 = 1;
			}
			else if ( LA1_0=='g' ) {
				alt1 = 2;
			}
		else {
			if (failed) return ;
		    ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:1 state:0 stream:input];
			@throw nvae;
			}
		}
		switch (alt1) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:25:6: grammarType= GRAMMAR_TYPE WS // alt
			    {
			    int _grammarTypeStart = [self charIndex];
			    [self mGRAMMAR_TYPE];
			    if (failed) return ;

			    _grammarType = [[ANTLRCommonToken alloc] initWithInput:input tokenType:ANTLRTokenTypeInvalid channel:ANTLRTokenChannelDefault start:_grammarTypeStart stop:[self charIndex]];
			    [_grammarType setLine:[self line]];
			    if ( backtracking==1 ) {
			      [delegate setGrammarType:[_grammarType text]]; 
			    }
			    [self mWS];
			    if (failed) return ;


			    }
			    break;
			case 2 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:26:53:  // alt
			    {
			    if ( backtracking==1 ) {
			      [delegate setGrammarType:@"parser"]; [delegate setIsCombinedGrammar:NO]; 
			    }

			    }
			    break;

		}

		[self matchString:@"grammar"];
		if (failed) return ;

		[self mWS];
		if (failed) return ;

		int _grammarNameStart = [self charIndex];
		[self mID];
		if (failed) return ;

		_grammarName = [[ANTLRCommonToken alloc] initWithInput:input tokenType:ANTLRTokenTypeInvalid channel:ANTLRTokenChannelDefault start:_grammarNameStart stop:[self charIndex]];
		[_grammarName setLine:[self line]];
		if ( backtracking==1 ) {
		   [delegate setGrammarName:[_grammarName text]]; 
		}
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:28:80: ( WS )? // block
		int alt2=2;
		{
			int LA2_0 = [input LA:1];
			if ( (LA2_0>='\t' && LA2_0<='\n')||LA2_0==' ' ) {
				alt2 = 1;
			}
		}
		switch (alt2) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:28:80: WS // alt
			    {
			    [self mWS];
			    if (failed) return ;


			    }
			    break;

		}

		[self matchChar:';'];
		if (failed) return ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		[_grammarType release];
		[_grammarName release];
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end GRAMMAR


- (void) mGRAMMAR_TYPE
{
	@try {
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:33:5: ( ( 'lexer' | 'parser' | 'tree' ) ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:33:5: ( 'lexer' | 'parser' | 'tree' ) // alt
		{
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:33:5: ( 'lexer' | 'parser' | 'tree' ) // block
		int alt3=3;
		switch ([input LA:1]) {
			case 'l':
				alt3 = 1;
				break;
			case 'p':
				alt3 = 2;
				break;
			case 't':
				alt3 = 3;
				break;
		default:
		 {
			if (failed) return ;
		    ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:3 state:0 stream:input];
			@throw nvae;

			}}
		switch (alt3) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:33:6: 'lexer' // alt
			    {
			    [self matchString:@"lexer"];
			    if (failed) return ;


			    }
			    break;
			case 2 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:33:14: 'parser' // alt
			    {
			    [self matchString:@"parser"];
			    if (failed) return ;


			    }
			    break;
			case 3 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:33:23: 'tree' // alt
			    {
			    [self matchString:@"tree"];
			    if (failed) return ;


			    }
			    break;

		}


		}

	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end GRAMMAR_TYPE


- (void) mOPTIONS
{
	ANTLRToken * _tokenVocab = nil;

	@try {
		int _type = GrammarFilterLexer_OPTIONS;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:38:4: ( 'options' ( WS )? '{' ( ( ( WS )? '//' )=> SL_COMMENT | ( ( WS )? '/*' )=> COMMENT | ( ( WS )? 'tokenVocab' )=> ( WS )? 'tokenVocab' ( WS )? '=' ( WS )? tokenVocab= ID ( WS )? ';' | ( WS )? ID ( WS )? '=' ( WS )? ID ( WS )? ';' )* ( WS )? '}' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:38:4: 'options' ( WS )? '{' ( ( ( WS )? '//' )=> SL_COMMENT | ( ( WS )? '/*' )=> COMMENT | ( ( WS )? 'tokenVocab' )=> ( WS )? 'tokenVocab' ( WS )? '=' ( WS )? tokenVocab= ID ( WS )? ';' | ( WS )? ID ( WS )? '=' ( WS )? ID ( WS )? ';' )* ( WS )? '}' // alt
		{
		[self matchString:@"options"];
		if (failed) return ;

		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:38:14: ( WS )? // block
		int alt4=2;
		{
			int LA4_0 = [input LA:1];
			if ( (LA4_0>='\t' && LA4_0<='\n')||LA4_0==' ' ) {
				alt4 = 1;
			}
		}
		switch (alt4) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:38:14: WS // alt
			    {
			    [self mWS];
			    if (failed) return ;


			    }
			    break;

		}

		[self matchChar:'{'];
		if (failed) return ;

		do {
		    int alt13=5;
		    alt13 = [dfa13 predict];
		    switch (alt13) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:39:5: ( ( WS )? '//' )=> SL_COMMENT // alt
			    {
			    [self mSL_COMMENT];
			    if (failed) return ;


			    }
			    break;
			case 2 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:40:5: ( ( WS )? '/*' )=> COMMENT // alt
			    {
			    [self mCOMMENT];
			    if (failed) return ;


			    }
			    break;
			case 3 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:5: ( ( WS )? 'tokenVocab' )=> ( WS )? 'tokenVocab' ( WS )? '=' ( WS )? tokenVocab= ID ( WS )? ';' // alt
			    {
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:27: ( WS )? // block
			    int alt5=2;
			    {
			    	int LA5_0 = [input LA:1];
			    	if ( (LA5_0>='\t' && LA5_0<='\n')||LA5_0==' ' ) {
			    		alt5 = 1;
			    	}
			    }
			    switch (alt5) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:27: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self matchString:@"tokenVocab"];
			    if (failed) return ;

			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:44: ( WS )? // block
			    int alt6=2;
			    {
			    	int LA6_0 = [input LA:1];
			    	if ( (LA6_0>='\t' && LA6_0<='\n')||LA6_0==' ' ) {
			    		alt6 = 1;
			    	}
			    }
			    switch (alt6) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:44: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self matchChar:'='];
			    if (failed) return ;

			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:52: ( WS )? // block
			    int alt7=2;
			    {
			    	int LA7_0 = [input LA:1];
			    	if ( (LA7_0>='\t' && LA7_0<='\n')||LA7_0==' ' ) {
			    		alt7 = 1;
			    	}
			    }
			    switch (alt7) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:52: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    int _tokenVocabStart = [self charIndex];
			    [self mID];
			    if (failed) return ;

			    _tokenVocab = [[ANTLRCommonToken alloc] initWithInput:input tokenType:ANTLRTokenTypeInvalid channel:ANTLRTokenChannelDefault start:_tokenVocabStart stop:[self charIndex]];
			    [_tokenVocab setLine:[self line]];
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:70: ( WS )? // block
			    int alt8=2;
			    {
			    	int LA8_0 = [input LA:1];
			    	if ( (LA8_0>='\t' && LA8_0<='\n')||LA8_0==' ' ) {
			    		alt8 = 1;
			    	}
			    }
			    switch (alt8) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:70: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self matchChar:';'];
			    if (failed) return ;

			    if ( backtracking==1 ) {
			       [delegate setDependsOnVocab:[_tokenVocab text]]; 
			    }

			    }
			    break;
			case 4 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:5: ( WS )? ID ( WS )? '=' ( WS )? ID ( WS )? ';' // alt
			    {
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:5: ( WS )? // block
			    int alt9=2;
			    {
			    	int LA9_0 = [input LA:1];
			    	if ( (LA9_0>='\t' && LA9_0<='\n')||LA9_0==' ' ) {
			    		alt9 = 1;
			    	}
			    }
			    switch (alt9) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:5: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self mID];
			    if (failed) return ;

			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:12: ( WS )? // block
			    int alt10=2;
			    {
			    	int LA10_0 = [input LA:1];
			    	if ( (LA10_0>='\t' && LA10_0<='\n')||LA10_0==' ' ) {
			    		alt10 = 1;
			    	}
			    }
			    switch (alt10) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:12: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self matchChar:'='];
			    if (failed) return ;

			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:20: ( WS )? // block
			    int alt11=2;
			    {
			    	int LA11_0 = [input LA:1];
			    	if ( (LA11_0>='\t' && LA11_0<='\n')||LA11_0==' ' ) {
			    		alt11 = 1;
			    	}
			    }
			    switch (alt11) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:20: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self mID];
			    if (failed) return ;

			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:27: ( WS )? // block
			    int alt12=2;
			    {
			    	int LA12_0 = [input LA:1];
			    	if ( (LA12_0>='\t' && LA12_0<='\n')||LA12_0==' ' ) {
			    		alt12 = 1;
			    	}
			    }
			    switch (alt12) {
			    	case 1 :
			    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:42:27: WS // alt
			    	    {
			    	    [self mWS];
			    	    if (failed) return ;


			    	    }
			    	    break;

			    }

			    [self matchChar:';'];
			    if (failed) return ;


			    }
			    break;

			default :
			    goto loop13;
		    }
		} while (YES); loop13: ;

		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:44:3: ( WS )? // block
		int alt14=2;
		{
			int LA14_0 = [input LA:1];
			if ( (LA14_0>='\t' && LA14_0<='\n')||LA14_0==' ' ) {
				alt14 = 1;
			}
		}
		switch (alt14) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:44:3: WS // alt
			    {
			    [self mWS];
			    if (failed) return ;


			    }
			    break;

		}

		[self matchChar:'}'];
		if (failed) return ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		[_tokenVocab release];
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end OPTIONS


- (void) mLEXER_RULE
{
	@try {
		int _type = GrammarFilterLexer_LEXER_RULE;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:4: ( ( 'A' .. 'Z' ) ( ID )? ( WS )? ':' ( options {greedy=false; } : . )* ';' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:4: ( 'A' .. 'Z' ) ( ID )? ( WS )? ':' ( options {greedy=false; } : . )* ';' // alt
		{
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:4: ( 'A' .. 'Z' ) // blockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:5: 'A' .. 'Z' // alt
		{
		[self matchRangeFromChar:'A' to:'Z'];if (failed) return ;

		}

		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:15: ( ID )? // block
		int alt15=2;
		{
			int LA15_0 = [input LA:1];
			if ( (LA15_0>='0' && LA15_0<='9')||(LA15_0>='A' && LA15_0<='Z')||LA15_0=='_'||(LA15_0>='a' && LA15_0<='z') ) {
				alt15 = 1;
			}
		}
		switch (alt15) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:15: ID // alt
			    {
			    [self mID];
			    if (failed) return ;


			    }
			    break;

		}

		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:19: ( WS )? // block
		int alt16=2;
		{
			int LA16_0 = [input LA:1];
			if ( (LA16_0>='\t' && LA16_0<='\n')||LA16_0==' ' ) {
				alt16 = 1;
			}
		}
		switch (alt16) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:19: WS // alt
			    {
			    [self mWS];
			    if (failed) return ;


			    }
			    break;

		}

		[self matchChar:':'];
		if (failed) return ;

		do {
		    int alt17=2;
		    {
		    	int LA17_0 = [input LA:1];
		    	if ( LA17_0==';' ) {
		    		alt17 = 2;
		    	}
		    	else if ( (LA17_0>=0x0000 && LA17_0<=':')||(LA17_0>='<' && LA17_0<=0xFFFE) ) {
		    		alt17 = 1;
		    	}

		    }
		    switch (alt17) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:49:54: . // alt
			    {
			    [self matchAny];
			    if (failed) return ;


			    }
			    break;

			default :
			    goto loop17;
		    }
		} while (YES); loop17: ;

		[self matchChar:';'];
		if (failed) return ;

		if ( backtracking==1 ) {
		   [delegate setIsCombinedGrammar:YES]; 
		}

		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end LEXER_RULE


- (void) mCOMMENT
{
	@try {
		int _type = GrammarFilterLexer_COMMENT;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:54:9: ( '/*' ( options {greedy=false; } : . )* '*/' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:54:9: '/*' ( options {greedy=false; } : . )* '*/' // alt
		{
		[self matchString:@"/*"];
		if (failed) return ;

		do {
		    int alt18=2;
		    {
		    	int LA18_0 = [input LA:1];
		    	if ( LA18_0=='*' ) {
		    		{
		    			int LA18_1 = [input LA:2];
		    			if ( LA18_1=='/' ) {
		    				alt18 = 2;
		    			}
		    			else if ( (LA18_1>=0x0000 && LA18_1<='.')||(LA18_1>='0' && LA18_1<=0xFFFE) ) {
		    				alt18 = 1;
		    			}

		    		}
		    	}
		    	else if ( (LA18_0>=0x0000 && LA18_0<=')')||(LA18_0>='+' && LA18_0<=0xFFFE) ) {
		    		alt18 = 1;
		    	}

		    }
		    switch (alt18) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:54:41: . // alt
			    {
			    [self matchAny];
			    if (failed) return ;


			    }
			    break;

			default :
			    goto loop18;
		    }
		} while (YES); loop18: ;

		[self matchString:@"*/"];
		if (failed) return ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end COMMENT


- (void) mSL_COMMENT
{
	@try {
		int _type = GrammarFilterLexer_SL_COMMENT;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:58:9: ( '//' ( options {greedy=false; } : . )* '\\n' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:58:9: '//' ( options {greedy=false; } : . )* '\\n' // alt
		{
		[self matchString:@"//"];
		if (failed) return ;

		do {
		    int alt19=2;
		    {
		    	int LA19_0 = [input LA:1];
		    	if ( LA19_0=='\n' ) {
		    		alt19 = 2;
		    	}
		    	else if ( (LA19_0>=0x0000 && LA19_0<='\t')||(LA19_0>=0x000B && LA19_0<=0xFFFE) ) {
		    		alt19 = 1;
		    	}

		    }
		    switch (alt19) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:58:41: . // alt
			    {
			    [self matchAny];
			    if (failed) return ;


			    }
			    break;

			default :
			    goto loop19;
		    }
		} while (YES); loop19: ;

		[self matchChar:'\n'];
		if (failed) return ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end SL_COMMENT


- (void) mACTION
{
	@try {
		int _type = GrammarFilterLexer_ACTION;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:63:4: ( '{' ( options {greedy=false; } : . )* '}' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:63:4: '{' ( options {greedy=false; } : . )* '}' // alt
		{
		[self matchChar:'{'];
		if (failed) return ;

		do {
		    int alt20=2;
		    {
		    	int LA20_0 = [input LA:1];
		    	if ( LA20_0=='}' ) {
		    		alt20 = 2;
		    	}
		    	else if ( (LA20_0>=0x0000 && LA20_0<='|')||(LA20_0>='~' && LA20_0<=0xFFFE) ) {
		    		alt20 = 1;
		    	}

		    }
		    switch (alt20) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:63:35: . // alt
			    {
			    [self matchAny];
			    if (failed) return ;


			    }
			    break;

			default :
			    goto loop20;
		    }
		} while (YES); loop20: ;

		[self matchChar:'}'];
		if (failed) return ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end ACTION


- (void) mSTRING
{
	@try {
		int _type = GrammarFilterLexer_STRING;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:67:4: ( '\\'' ( options {greedy=false; } : . )* '\\'' ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:67:4: '\\'' ( options {greedy=false; } : . )* '\\'' // alt
		{
		[self matchChar:'\''];
		if (failed) return ;

		do {
		    int alt21=2;
		    {
		    	int LA21_0 = [input LA:1];
		    	if ( LA21_0=='\'' ) {
		    		alt21 = 2;
		    	}
		    	else if ( (LA21_0>=0x0000 && LA21_0<='&')||(LA21_0>='(' && LA21_0<=0xFFFE) ) {
		    		alt21 = 1;
		    	}

		    }
		    switch (alt21) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:67:36: . // alt
			    {
			    [self matchAny];
			    if (failed) return ;


			    }
			    break;

			default :
			    goto loop21;
		    }
		} while (YES); loop21: ;

		[self matchChar:'\''];
		if (failed) return ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end STRING


- (void) mID
{
	@try {
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:71:9: ( ( ('a'..'z'|'A'..'Z'|'_'|'0'..'9'))+ ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:71:9: ( ('a'..'z'|'A'..'Z'|'_'|'0'..'9'))+ // alt
		{
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:71:9: ( ('a'..'z'|'A'..'Z'|'_'|'0'..'9'))+	// positiveClosureBlock
		int cnt22=0;

		do {
		    int alt22=2;
		    {
		    	int LA22_0 = [input LA:1];
		    	if ( (LA22_0>='0' && LA22_0<='9')||(LA22_0>='A' && LA22_0<='Z')||LA22_0=='_'||(LA22_0>='a' && LA22_0<='z') ) {
		    		alt22 = 1;
		    	}

		    }
		    switch (alt22) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:71:10: ('a'..'z'|'A'..'Z'|'_'|'0'..'9') // alt
			    {
			    if (([input LA:1]>='0' && [input LA:1]<='9')||([input LA:1]>='A' && [input LA:1]<='Z')||[input LA:1]=='_'||([input LA:1]>='a' && [input LA:1]<='z')) {
			    	[input consume];
			    failed = NO;
			    } else {
			    	if (backtracking > 0) {failed=YES; return ;}
			    	ANTLRMismatchedSetException *mse = [ANTLRMismatchedSetException exceptionWithSet:nil stream:input];
			    	[self recover:mse];	@throw mse;
			    }


			    }
			    break;

			default :
			    if ( cnt22 >= 1 )  goto loop22;
		            if (backtracking > 0) {failed=YES; return ;}
					ANTLREarlyExitException *eee = [ANTLREarlyExitException exceptionWithStream:input decisionNumber:22];
					@throw eee;
		    }
		    cnt22++;
		} while (YES); loop22: ;


		}

	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end ID


- (void) mWS
{
	@try {
		int _type = GrammarFilterLexer_WS;
		int _start = [self charIndex];
		int _line = [self line];
		int _charPosition = [self charPositionInLine];
		int _channel = [ANTLRToken defaultChannel];
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:74:9: ( ( (' '|'\\t'|'\\n'))+ ) // ruleBlockSingleAlt
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:74:9: ( (' '|'\\t'|'\\n'))+ // alt
		{
		// /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:74:9: ( (' '|'\\t'|'\\n'))+	// positiveClosureBlock
		int cnt23=0;

		do {
		    int alt23=2;
		    {
		    	int LA23_0 = [input LA:1];
		    	if ( (LA23_0>='\t' && LA23_0<='\n')||LA23_0==' ' ) {
		    		alt23 = 1;
		    	}

		    }
		    switch (alt23) {
			case 1 :
			    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:74:10: (' '|'\\t'|'\\n') // alt
			    {
			    if (([input LA:1]>='\t' && [input LA:1]<='\n')||[input LA:1]==' ') {
			    	[input consume];
			    failed = NO;
			    } else {
			    	if (backtracking > 0) {failed=YES; return ;}
			    	ANTLRMismatchedSetException *mse = [ANTLRMismatchedSetException exceptionWithSet:nil stream:input];
			    	[self recover:mse];	@throw mse;
			    }


			    }
			    break;

			default :
			    if ( cnt23 >= 1 )  goto loop23;
		            if (backtracking > 0) {failed=YES; return ;}
					ANTLREarlyExitException *eee = [ANTLREarlyExitException exceptionWithStream:input decisionNumber:23];
					@throw eee;
		    }
		    cnt23++;
		} while (YES); loop23: ;


		}

		if ( token == nil ) { [self emitTokenWithType:_type line:_line charPosition:_charPosition channel:_channel start:_start stop:[self charIndex]];}
	}
	@finally {
        // rule cleanup
		// token labels
		// token+rule list labels
		// rule labels
		// rule refs in alts with rewrites

		if ( backtracking == 0 ) {
		}
	}
	return;
}
// $ANTLR end WS

- (void) mTokens
{
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:25: ( ( GRAMMAR )=> GRAMMAR | ( OPTIONS )=> OPTIONS | ( LEXER_RULE )=> LEXER_RULE | ( COMMENT )=> COMMENT | ( SL_COMMENT )=> SL_COMMENT | ( ACTION )=> ACTION | ( STRING )=> STRING | ( WS )=> WS ) //ruleblock
    int alt24=8;
    switch ([input LA:1]) {
    	case 'g':
    	case 'l':
    	case 'p':
    	case 't':
    		alt24 = 1;
    		break;
    	case 'o':
    		alt24 = 2;
    		break;
    	case 'A':
    	case 'B':
    	case 'C':
    	case 'D':
    	case 'E':
    	case 'F':
    	case 'G':
    	case 'H':
    	case 'I':
    	case 'J':
    	case 'K':
    	case 'L':
    	case 'M':
    	case 'N':
    	case 'O':
    	case 'P':
    	case 'Q':
    	case 'R':
    	case 'S':
    	case 'T':
    	case 'U':
    	case 'V':
    	case 'W':
    	case 'X':
    	case 'Y':
    	case 'Z':
    		alt24 = 3;
    		break;
    	case '/':
    		{
    			int LA24_7 = [input LA:2];
    			if ( [self evaluateSyntacticPredicate:@selector(synpred7)] ) {
    				alt24 = 4;
    			}
    			else if ( [self evaluateSyntacticPredicate:@selector(synpred8)] ) {
    				alt24 = 5;
    			}
    		else {
    			if (failed) return ;
    		    ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:24 state:7 stream:input];
    			@throw nvae;
    			}
    		}
    		break;
    	case '{':
    		alt24 = 6;
    		break;
    	case '\'':
    		alt24 = 7;
    		break;
    	case '\t':
    	case '\n':
    	case ' ':
    		alt24 = 8;
    		break;
    default:
     {
    	if (failed) return ;
        ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:24 state:0 stream:input];
    	@throw nvae;

    	}}
    switch (alt24) {
    	case 1 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:25: ( GRAMMAR )=> GRAMMAR // alt
    	    {
    	    [self mGRAMMAR];
    	    if (failed) return ;


    	    }
    	    break;
    	case 2 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:44: ( OPTIONS )=> OPTIONS // alt
    	    {
    	    [self mOPTIONS];
    	    if (failed) return ;


    	    }
    	    break;
    	case 3 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:63: ( LEXER_RULE )=> LEXER_RULE // alt
    	    {
    	    [self mLEXER_RULE];
    	    if (failed) return ;


    	    }
    	    break;
    	case 4 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:88: ( COMMENT )=> COMMENT // alt
    	    {
    	    [self mCOMMENT];
    	    if (failed) return ;


    	    }
    	    break;
    	case 5 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:107: ( SL_COMMENT )=> SL_COMMENT // alt
    	    {
    	    [self mSL_COMMENT];
    	    if (failed) return ;


    	    }
    	    break;
    	case 6 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:132: ( ACTION )=> ACTION // alt
    	    {
    	    [self mACTION];
    	    if (failed) return ;


    	    }
    	    break;
    	case 7 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:149: ( STRING )=> STRING // alt
    	    {
    	    [self mSTRING];
    	    if (failed) return ;


    	    }
    	    break;
    	case 8 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:166: ( WS )=> WS // alt
    	    {
    	    [self mWS];
    	    if (failed) return ;


    	    }
    	    break;

    }

}

- (void) synpred3
{
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:5: ( ( WS )? 'tokenVocab' ) // ruleBlockSingleAlt
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:6: ( WS )? 'tokenVocab' // alt
    {
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:6: ( WS )? // block
    int alt27=2;
    {
    	int LA27_0 = [input LA:1];
    	if ( (LA27_0>='\t' && LA27_0<='\n')||LA27_0==' ' ) {
    		alt27 = 1;
    	}
    }
    switch (alt27) {
    	case 1 :
    	    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:41:6: WS // alt
    	    {
    	    [self mWS];
    	    if (failed) return ;


    	    }
    	    break;

    }

    [self matchString:@"tokenVocab"];
    if (failed) return ;


    }
}

- (void) synpred7
{
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:88: ( COMMENT ) // ruleBlockSingleAlt
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:89: COMMENT // alt
    {
    [self mCOMMENT];
    if (failed) return ;


    }
}

- (void) synpred8
{
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:107: ( SL_COMMENT ) // ruleBlockSingleAlt
    // /Users/kroepke/Projects/antlr3/code/antlr/main/lib/ObjC/Xcode plugin/GrammarFilter.g:1:108: SL_COMMENT // alt
    {
    [self mSL_COMMENT];
    if (failed) return ;


    }
}

@end