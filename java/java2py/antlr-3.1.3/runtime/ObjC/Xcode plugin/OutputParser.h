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

@protocol XCOutputStreams <NSObject>
- (void)writeBytes:(const char *)fp8 length:(unsigned int)fp12;
- (void)flush;
- (void)close;
@end

@interface XCOutputStream : NSObject <XCOutputStreams>
{
}

- (void)writeBytes:(const char *)fp8 length:(unsigned int)fp12;
- (void)flush;
- (void)close;
- (void)writeData:(id)fp8;

@end

@interface XCFileOutputStream : XCOutputStream
{
    int _fileDescriptor;
    BOOL _closeFDWhenStreamIsClosed;
}

+ (id)stdoutFileOutputStream;
+ (id)stderrFileOutputStream;
+ (id)nullFileOutputStream;
- (id)initWithFileDescriptor:(int)fp8 closeFileDescriptorWhenStreamIsClosed:(BOOL)fp12;
- (id)init;
- (void)dealloc;
- (void)finalize;
- (void)writeBytes:(const char *)fp8 length:(unsigned int)fp12;
- (void)flush;
- (void)close;

@end


@interface XCFilterOutputStream : XCOutputStream
{
    id _nextOutputStream;
}

- (id)initWithNextOutputStream:(id)fp8;
- (id)init;
- (void)dealloc;
- (id)nextOutputStream;
- (void)setNextOutputStream:(id)fp8;
- (id)lastOutputStream;
- (void)writeBytes:(const char *)fp8 length:(unsigned int)fp12;
- (void)flush;
- (void)close;

@end


@interface XCBuildCommandOutputParser : XCFilterOutputStream
{
    id _delegate;
}

- (id)initWithNextOutputStream:(id)fp8;
- (id)delegate;
- (void)setDelegate:(id)fp8;
- (void)writeBytes:(const char *)fp8 length:(unsigned int)fp12;

@end

