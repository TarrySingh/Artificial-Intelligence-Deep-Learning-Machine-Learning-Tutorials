//
//  TestStringStream.m
//  ANTLR
//
//  Created by Kay Röpke on 09.03.2006.
//  Copyright 2006 classDump. All rights reserved.
//

#import "TestStringStream.h"
#import "ANTLRStringStream.h"

@implementation TestStringStream

- (void) testStringStreamCreation
{
	NSString *input = @"This is a sample input string.";
	ANTLRStringStream *stringStream = [[ANTLRStringStream alloc] initWithStringNoCopy:input];
	
	NSString *substr = [stringStream substringWithRange:NSMakeRange(0,10)];
	NSLog(@"the first 10 chars are: %@", substr);
	STAssertTrue([@"This is a " isEqualToString:substr], @"The substrings are not equal. Got: %@", substr);
}

@end
