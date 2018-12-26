// $ANTLR 3.0 SymbolTable.g 2007-07-25 20:12:44

#import "SymbolTableParser.h"


#pragma mark Bitsets
const static unsigned long long FOLLOW_globals_in_prog44_data[] = {0x0000000000000082LL};
static ANTLRBitSet *FOLLOW_globals_in_prog44;
const static unsigned long long FOLLOW_method_in_prog47_data[] = {0x0000000000000082LL};
static ANTLRBitSet *FOLLOW_method_in_prog47;
const static unsigned long long FOLLOW_decl_in_globals79_data[] = {0x0000000000004002LL};
static ANTLRBitSet *FOLLOW_decl_in_globals79;
const static unsigned long long FOLLOW_7_in_method110_data[] = {0x0000000000000010LL};
static ANTLRBitSet *FOLLOW_7_in_method110;
const static unsigned long long FOLLOW_SymbolTableParser_ID_in_method112_data[] = {0x0000000000000100LL};
static ANTLRBitSet *FOLLOW_SymbolTableParser_ID_in_method112;
const static unsigned long long FOLLOW_8_in_method114_data[] = {0x0000000000000200LL};
static ANTLRBitSet *FOLLOW_8_in_method114;
const static unsigned long long FOLLOW_9_in_method116_data[] = {0x0000000000000400LL};
static ANTLRBitSet *FOLLOW_9_in_method116;
const static unsigned long long FOLLOW_block_in_method118_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_block_in_method118;
const static unsigned long long FOLLOW_10_in_block147_data[] = {0x0000000000004C10LL};
static ANTLRBitSet *FOLLOW_10_in_block147;
const static unsigned long long FOLLOW_decl_in_block150_data[] = {0x0000000000004C10LL};
static ANTLRBitSet *FOLLOW_decl_in_block150;
const static unsigned long long FOLLOW_stat_in_block155_data[] = {0x0000000000000C10LL};
static ANTLRBitSet *FOLLOW_stat_in_block155;
const static unsigned long long FOLLOW_11_in_block159_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_11_in_block159;
const static unsigned long long FOLLOW_SymbolTableParser_ID_in_stat183_data[] = {0x0000000000001000LL};
static ANTLRBitSet *FOLLOW_SymbolTableParser_ID_in_stat183;
const static unsigned long long FOLLOW_12_in_stat185_data[] = {0x0000000000000020LL};
static ANTLRBitSet *FOLLOW_12_in_stat185;
const static unsigned long long FOLLOW_SymbolTableParser_INT_in_stat187_data[] = {0x0000000000002000LL};
static ANTLRBitSet *FOLLOW_SymbolTableParser_INT_in_stat187;
const static unsigned long long FOLLOW_13_in_stat189_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_13_in_stat189;
const static unsigned long long FOLLOW_block_in_stat199_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_block_in_stat199;
const static unsigned long long FOLLOW_14_in_decl213_data[] = {0x0000000000000010LL};
static ANTLRBitSet *FOLLOW_14_in_decl213;
const static unsigned long long FOLLOW_SymbolTableParser_ID_in_decl215_data[] = {0x0000000000002000LL};
static ANTLRBitSet *FOLLOW_SymbolTableParser_ID_in_decl215;
const static unsigned long long FOLLOW_13_in_decl217_data[] = {0x0000000000000002LL};
static ANTLRBitSet *FOLLOW_13_in_decl217;


#pragma mark Dynamic Global Scopes
@implementation SymbolTableParserSymbolsScope
@end

#pragma mark Dynamic Rule Scopes

#pragma mark Rule return scopes start

@implementation SymbolTableParser

+ (void) initialize
{
	FOLLOW_globals_in_prog44 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_globals_in_prog44_data count:1];
	FOLLOW_method_in_prog47 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_method_in_prog47_data count:1];
	FOLLOW_decl_in_globals79 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_decl_in_globals79_data count:1];
	FOLLOW_7_in_method110 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_7_in_method110_data count:1];
	FOLLOW_SymbolTableParser_ID_in_method112 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_SymbolTableParser_ID_in_method112_data count:1];
	FOLLOW_8_in_method114 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_8_in_method114_data count:1];
	FOLLOW_9_in_method116 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_9_in_method116_data count:1];
	FOLLOW_block_in_method118 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_block_in_method118_data count:1];
	FOLLOW_10_in_block147 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_10_in_block147_data count:1];
	FOLLOW_decl_in_block150 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_decl_in_block150_data count:1];
	FOLLOW_stat_in_block155 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_stat_in_block155_data count:1];
	FOLLOW_11_in_block159 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_11_in_block159_data count:1];
	FOLLOW_SymbolTableParser_ID_in_stat183 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_SymbolTableParser_ID_in_stat183_data count:1];
	FOLLOW_12_in_stat185 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_12_in_stat185_data count:1];
	FOLLOW_SymbolTableParser_INT_in_stat187 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_SymbolTableParser_INT_in_stat187_data count:1];
	FOLLOW_13_in_stat189 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_13_in_stat189_data count:1];
	FOLLOW_block_in_stat199 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_block_in_stat199_data count:1];
	FOLLOW_14_in_decl213 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_14_in_decl213_data count:1];
	FOLLOW_SymbolTableParser_ID_in_decl215 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_SymbolTableParser_ID_in_decl215_data count:1];
	FOLLOW_13_in_decl217 = [[ANTLRBitSet alloc] initWithBits:FOLLOW_13_in_decl217_data count:1];

}

- (id) initWithTokenStream:(id<ANTLRTokenStream>)aStream
{
	if ((self = [super initWithTokenStream:aStream])) {

		tokenNames = [[NSArray alloc] initWithObjects:@"<invalid>", @"<EOR>", @"<DOWN>", @"<UP>", 
	@"ID", @"INT", @"WS", @"'method'", @"'('", @"')'", @"'{'", @"'}'", @"'='", 
	@"';'", @"'int'", nil];

		SymbolTableParser_Symbols_stack = [[NSMutableArray alloc] init];
														

		level = 0;

	}
	return self;
}

- (void) dealloc
{
	[tokenNames release];

	[SymbolTableParser_Symbols_stack release];
	[super dealloc];
}

- (NSString *) grammarFileName
{
	return @"SymbolTable.g";
}


// $ANTLR start prog
// SymbolTable.g:25:1: prog : globals ( method )* ;
- (void) prog
{
    @try {
        // SymbolTable.g:25:9: ( globals ( method )* ) // ruleBlockSingleAlt
        // SymbolTable.g:25:9: globals ( method )* // alt
        {
        [following addObject:FOLLOW_globals_in_prog44];
        [self globals];
        [following removeLastObject];


        do {
            int alt1=2;
            {
            	int LA1_0 = [input LA:1];
            	if ( LA1_0==7 ) {
            		alt1 = 1;
            	}

            }
            switch (alt1) {
        	case 1 :
        	    // SymbolTable.g:25:18: method // alt
        	    {
        	    [following addObject:FOLLOW_method_in_prog47];
        	    [self method];
        	    [following removeLastObject];



        	    }
        	    break;

        	default :
        	    goto loop1;
            }
        } while (YES); loop1: ;


        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end prog

// $ANTLR start globals
// SymbolTable.g:28:1: globals : ( decl )* ;
- (void) globals
{
    [SymbolTableParser_Symbols_stack addObject:[[[SymbolTableParserSymbolsScope alloc] init] autorelease]];


        level++;
        [[SymbolTableParser_Symbols_stack lastObject] setValue: [[NSMutableArray alloc] init] forKey:@"names"];

    @try {
        // SymbolTable.g:34:9: ( ( decl )* ) // ruleBlockSingleAlt
        // SymbolTable.g:34:9: ( decl )* // alt
        {
        do {
            int alt2=2;
            {
            	int LA2_0 = [input LA:1];
            	if ( LA2_0==14 ) {
            		alt2 = 1;
            	}

            }
            switch (alt2) {
        	case 1 :
        	    // SymbolTable.g:34:10: decl // alt
        	    {
        	    [following addObject:FOLLOW_decl_in_globals79];
        	    [self decl];
        	    [following removeLastObject];



        	    }
        	    break;

        	default :
        	    goto loop2;
            }
        } while (YES); loop2: ;


                NSLog(@"globals: %@", [[SymbolTableParser_Symbols_stack lastObject] valueForKey:@"names"]);
                level--;
                

        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

		[SymbolTableParser_Symbols_stack removeLastObject];

	}
	return ;
}
// $ANTLR end globals

// $ANTLR start method
// SymbolTable.g:41:1: method : 'method' ID '(' ')' block ;
- (void) method
{
    @try {
        // SymbolTable.g:42:9: ( 'method' ID '(' ')' block ) // ruleBlockSingleAlt
        // SymbolTable.g:42:9: 'method' ID '(' ')' block // alt
        {
        [self match:input tokenType:7 follow:FOLLOW_7_in_method110]; 
        [self match:input tokenType:SymbolTableParser_ID follow:FOLLOW_SymbolTableParser_ID_in_method112]; 
        [self match:input tokenType:8 follow:FOLLOW_8_in_method114]; 
        [self match:input tokenType:9 follow:FOLLOW_9_in_method116]; 
        [following addObject:FOLLOW_block_in_method118];
        [self block];
        [following removeLastObject];



        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end method

// $ANTLR start block
// SymbolTable.g:45:1: block : '{' ( decl )* ( stat )* '}' ;
- (void) block
{
    [SymbolTableParser_Symbols_stack addObject:[[[SymbolTableParserSymbolsScope alloc] init] autorelease]];


        level++;
        [[SymbolTableParser_Symbols_stack lastObject] setValue: [[NSMutableArray alloc] init] forKey:@"names"];

    @try {
        // SymbolTable.g:51:9: ( '{' ( decl )* ( stat )* '}' ) // ruleBlockSingleAlt
        // SymbolTable.g:51:9: '{' ( decl )* ( stat )* '}' // alt
        {
        [self match:input tokenType:10 follow:FOLLOW_10_in_block147]; 
        do {
            int alt3=2;
            {
            	int LA3_0 = [input LA:1];
            	if ( LA3_0==14 ) {
            		alt3 = 1;
            	}

            }
            switch (alt3) {
        	case 1 :
        	    // SymbolTable.g:51:14: decl // alt
        	    {
        	    [following addObject:FOLLOW_decl_in_block150];
        	    [self decl];
        	    [following removeLastObject];



        	    }
        	    break;

        	default :
        	    goto loop3;
            }
        } while (YES); loop3: ;

        do {
            int alt4=2;
            {
            	int LA4_0 = [input LA:1];
            	if ( LA4_0==SymbolTableParser_ID||LA4_0==10 ) {
            		alt4 = 1;
            	}

            }
            switch (alt4) {
        	case 1 :
        	    // SymbolTable.g:51:22: stat // alt
        	    {
        	    [following addObject:FOLLOW_stat_in_block155];
        	    [self stat];
        	    [following removeLastObject];



        	    }
        	    break;

        	default :
        	    goto loop4;
            }
        } while (YES); loop4: ;

        [self match:input tokenType:11 follow:FOLLOW_11_in_block159]; 

                NSLog(@"level %d symbols: %@", level, [[SymbolTableParser_Symbols_stack lastObject] valueForKey:@"names"]);
                level--;
                

        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

		[SymbolTableParser_Symbols_stack removeLastObject];

	}
	return ;
}
// $ANTLR end block

// $ANTLR start stat
// SymbolTable.g:58:1: stat : ( ID '=' INT ';' | block );
- (void) stat
{
    @try {
        // SymbolTable.g:58:9: ( ID '=' INT ';' | block ) //ruleblock
        int alt5=2;
        {
        	int LA5_0 = [input LA:1];
        	if ( LA5_0==SymbolTableParser_ID ) {
        		alt5 = 1;
        	}
        	else if ( LA5_0==10 ) {
        		alt5 = 2;
        	}
        else {
            ANTLRNoViableAltException *nvae = [ANTLRNoViableAltException exceptionWithDecision:5 state:0 stream:input];
        	@throw nvae;
        	}
        }
        switch (alt5) {
        	case 1 :
        	    // SymbolTable.g:58:9: ID '=' INT ';' // alt
        	    {
        	    [self match:input tokenType:SymbolTableParser_ID follow:FOLLOW_SymbolTableParser_ID_in_stat183]; 
        	    [self match:input tokenType:12 follow:FOLLOW_12_in_stat185]; 
        	    [self match:input tokenType:SymbolTableParser_INT follow:FOLLOW_SymbolTableParser_INT_in_stat187]; 
        	    [self match:input tokenType:13 follow:FOLLOW_13_in_stat189]; 

        	    }
        	    break;
        	case 2 :
        	    // SymbolTable.g:59:9: block // alt
        	    {
        	    [following addObject:FOLLOW_block_in_stat199];
        	    [self block];
        	    [following removeLastObject];



        	    }
        	    break;

        }
    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end stat

// $ANTLR start decl
// SymbolTable.g:62:1: decl : 'int' ID ';' ;
- (void) decl
{
    id<ANTLRToken>  _ID1 = nil;

    @try {
        // SymbolTable.g:62:9: ( 'int' ID ';' ) // ruleBlockSingleAlt
        // SymbolTable.g:62:9: 'int' ID ';' // alt
        {
        [self match:input tokenType:14 follow:FOLLOW_14_in_decl213]; 
        _ID1=(id<ANTLRToken> )[input LT:1];
        [_ID1 retain];
        [self match:input tokenType:SymbolTableParser_ID follow:FOLLOW_SymbolTableParser_ID_in_decl215]; 
        [self match:input tokenType:13 follow:FOLLOW_13_in_decl217]; 
        [[[SymbolTableParser_Symbols_stack lastObject] valueForKey:@"names"] addObject:_ID1];

        }

    }
	@catch (ANTLRRecognitionException *re) {
		[self reportError:re];
		[self recover:input exception:re];
	}
	@finally {
		// token labels
		[_ID1 release];
		// token+rule list labels
		// rule labels

	}
	return ;
}
// $ANTLR end decl



@end