package org.antlr.runtime.test {
	import flexunit.framework.TestCase;
	
	import org.antlr.runtime.DFA;
	
	public class TestDFA extends TestCase {
	
		public function testUnpack():void {
			// empty
			var testVal:String = "\x01\x02\x03\x09";
			assertEquals(4, testVal.length);
			assertEquals("2,9,9,9", DFA.unpackEncodedString("\x01\x02\x03\x09"));
			
			testVal = "\x03\u7fff";
			//testVal = String.fromCharCode(3, 0x7fff);
			
			assertEquals(2, testVal.length);
			assertEquals("32767,32767,32767", DFA.unpackEncodedString(testVal));
			assertEquals("32767,32767,32767", DFA.unpackEncodedString(testVal, true));
			
			testVal = "\x02\u80ff\xff";
			assertEquals(3, testVal.length);
			assertEquals("-1,-1", DFA.unpackEncodedString(testVal));
			assertEquals("65535,65535", DFA.unpackEncodedString(testVal, true));

		}	

	}
}