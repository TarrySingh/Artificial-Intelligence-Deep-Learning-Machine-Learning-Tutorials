package org.antlr.gunit;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for gUnit itself....
 */
public class GunitTest
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public GunitTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( GunitTest.class );
    }

    /**
     * Rigourous Test :-)
     */
    public void testGunitTest()
    {
        assertTrue( true );
    }
}
