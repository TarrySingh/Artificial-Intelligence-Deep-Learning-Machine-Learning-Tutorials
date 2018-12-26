package org.antlr.runtime.test {
	import flexunit.framework.TestCase;
	
	import org.antlr.runtime.ANTLRStringStream;
	import org.antlr.runtime.CharStream;
	import org.antlr.runtime.CharStreamConstants;
	
	public class TestANTLRStringStream extends TestCase {
		
		public function TestANTLRStringStream()	{
			super();
		}
		
		public function testConsume():void {
			var stream:CharStream = new ANTLRStringStream("abc");
			assertEquals(stream.size, 3);
			assertEquals(stream.charPositionInLine, 0);
			assertEquals(stream.line, 1);
			assertEquals(stream.index, 0);
			
			for (var i:int = 0; i < stream.size; i++) {
				stream.consume();
				
				assertEquals(stream.size, 3); // invariant
				assertEquals(stream.charPositionInLine, i + 1);  
				assertEquals(stream.line, 1); // invariant
				assertEquals(stream.index, i + 1);
			}
			
			// now consume past EOF for a few ticks, nothing should change
			for (i = 0; i < 5; i++) {
				stream.consume();
				
				assertEquals(stream.size, 3); // invariant
				assertEquals(stream.charPositionInLine, 3);  
				assertEquals(stream.line, 1); // invariant
				assertEquals(stream.index, 3);
			}
				
		}
		
		public function testLA():void {
			var stream:CharStream = new ANTLRStringStream("abc");
			assertEquals(stream.LA(-1), CharStreamConstants.EOF);  // should be EOF
			assertEquals(stream.LA(0), 0); // should be 0 (undefined)
			assertEquals(stream.LA(1), "a".charCodeAt(0));
			assertEquals(stream.LA(2), "b".charCodeAt(0));
			assertEquals(stream.LA(3), "c".charCodeAt(0));
			assertEquals(stream.LA(4), CharStreamConstants.EOF);
			
			// now consume() one byte and run some more tests.
			stream.consume();
			assertEquals(stream.LA(-2), CharStreamConstants.EOF);
			assertEquals(stream.LA(-1), "a".charCodeAt(0));
			assertEquals(stream.LA(0), 0); // should be 0 (undefined)
			assertEquals(stream.LA(1), "b".charCodeAt(0));
			assertEquals(stream.LA(2), "c".charCodeAt(0));
			assertEquals(stream.LA(3), CharStreamConstants.EOF);
		}
		
		public function testReset():void {
			var stream:ANTLRStringStream = new ANTLRStringStream("abc");
			assertEquals(stream.size, 3);
			assertEquals(stream.charPositionInLine, 0);
			assertEquals(stream.line, 1);
			assertEquals(stream.index, 0);
			
			stream.consume();
			stream.consume();
			
			assertEquals(stream.size, 3);
			assertEquals(stream.charPositionInLine, 2);
			assertEquals(stream.line, 1);
			assertEquals(stream.index, 2);
			
			stream.reset();
			
			assertEquals(stream.size, 3);
			assertEquals(stream.charPositionInLine, 0);
			assertEquals(stream.line, 1);
			assertEquals(stream.index, 0);
			
		}
		
		public function testMark():void {
			var stream:ANTLRStringStream = new ANTLRStringStream("a\nbc");
			
			// setup a couple of markers
			var mark1:int = stream.mark();
			stream.consume();
			stream.consume();
			var mark2:int = stream.mark();
			stream.consume();
			
			// make sure we are where we expect to be
			assertEquals(stream.charPositionInLine, 1);
			assertEquals(stream.line, 2);
			assertEquals(stream.index, 3);
			
			assertEquals(mark1, 1);
			assertTrue(mark1 != mark2);
			
			stream.rewindTo(mark2);
			assertEquals(stream.charPositionInLine, 0);
			assertEquals(stream.line, 2);
			assertEquals(stream.index, 2);
			
			stream.rewindTo(mark1);
			assertEquals(stream.index, 0);
			assertEquals(stream.charPositionInLine, 0);
			assertEquals(stream.line, 1);
			
			// test two-level rewind
			mark1 = stream.mark();
			stream.consume();
			stream.consume();
			stream.mark();
			stream.consume();
			
			// make sure we are where we expect to be
			assertEquals(stream.charPositionInLine, 1);
			assertEquals(stream.line, 2);
			assertEquals(stream.index, 3);
			
			stream.rewindTo(mark1);
			assertEquals(stream.index, 0);
			assertEquals(stream.charPositionInLine, 0);
			assertEquals(stream.line, 1);	
		}
	}
}