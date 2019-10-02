package org.antlr.runtime.test {
	import flexunit.framework.TestCase;
	
	import org.antlr.runtime.BitSet;
	
	public class TestBitSet extends TestCase {
	
		public function testConstructor():void {
			// empty
			var bitSet:BitSet = new BitSet();
			
			assertEquals(0, bitSet.numBits);
			assertEquals(0, bitSet.toPackedArray().length);
			assertEquals(0, bitSet.size);
			assertTrue(bitSet.isNil);
			assertEquals("{}", bitSet.toString());
			
			bitSet = BitSet.of(0, 1, 2);
			assertEquals(32, bitSet.numBits);
			assertEquals(1, bitSet.toPackedArray().length);
			//assertEquals(1, bitSet.size);
			assertFalse(bitSet.isNil);
			assertEquals(7, int(bitSet.toPackedArray()[0]));
			assertEquals("{0,1,2}", bitSet.toString());
			
			
		}	

	}
}