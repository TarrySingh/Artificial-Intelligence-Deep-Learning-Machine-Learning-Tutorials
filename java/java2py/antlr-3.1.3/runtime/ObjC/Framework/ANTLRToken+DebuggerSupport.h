//
//  ANTLRToken+DebuggerSupport.h
//  ANTLR
//
//  Created by Kay Röpke on 03.12.2006.
//  Copyright 2006 classDump. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRCommonToken.h>

@interface ANTLRCommonToken(DebuggerSupport)

- (NSString *)debuggerDescription;

@end
