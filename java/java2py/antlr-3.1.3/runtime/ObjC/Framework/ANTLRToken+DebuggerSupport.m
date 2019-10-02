//
//  ANTLRToken+DebuggerSupport.m
//  ANTLR
//
//  Created by Kay Röpke on 03.12.2006.
//  Copyright 2006 classDump. All rights reserved.
//

#import "ANTLRToken+DebuggerSupport.h"


@implementation ANTLRCommonToken(DebuggerSupport)

- (NSString *)debuggerDescription
{
	NSString *_text = [self text];
	NSMutableString *escapedText;
	if (_text) {
		escapedText = [_text mutableCopy];
		NSRange wholeString = NSMakeRange(0,[escapedText length]);
		[escapedText replaceOccurrencesOfString:@"%" withString:@"%25" options:0 range:wholeString];
		[escapedText replaceOccurrencesOfString:@"\n" withString:@"%0A" options:0 range:wholeString];
		[escapedText replaceOccurrencesOfString:@"\r" withString:@"%0D" options:0 range:wholeString];
	} else {
		escapedText = [NSMutableString stringWithString:@""];
	}
	// format is tokenIndex, type, channel, line, col, (escaped)text
	return [NSString stringWithFormat:@"%u %d %u %u %u \"%@", 
		[self tokenIndex],
		[self type],
		[self channel],
		[self line],
		[self charPositionInLine],
		escapedText
		];
}

@end
