package org.antlr.runtime.test {
	import flexunit.framework.TestSuite;
	
	
	public class AllTests extends TestSuite {
		
		public function AllTests() {
			addTest(new TestSuite(TestANTLRStringStream));
			addTest(new TestSuite(TestBitSet));
			addTest(new TestSuite(TestDFA));
		}
		
	}
}