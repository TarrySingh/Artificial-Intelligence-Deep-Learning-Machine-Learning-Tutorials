package org.antlr.codegen;
import java.util.*;

public class JavaScriptTarget extends Target {
    /** Convert an int to a JavaScript Unicode character literal.
     *
     *  The current JavaScript spec (ECMA-262) doesn't provide for octal
     *  notation in String literals, although some implementations support it.
     *  This method overrides the parent class so that characters will always
     *  be encoded as Unicode literals (e.g. \u0011).
     */
    public String encodeIntAsCharEscape(int v) {
        String hex = Integer.toHexString(v|0x10000).substring(1,5);
        return "\\u"+hex;
    }

    /** Convert long to two 32-bit numbers separted by a comma.
     *  JavaScript does not support 64-bit numbers, so we need to break
     *  the number into two 32-bit literals to give to the Bit.  A number like
     *  0xHHHHHHHHLLLLLLLL is broken into the following string:
     *  "0xLLLLLLLL, 0xHHHHHHHH"
     *  Note that the low order bits are first, followed by the high order bits.
     *  This is to match how the BitSet constructor works, where the bits are
     *  passed in in 32-bit chunks with low-order bits coming first.
     *
     *  Note: stole the following two methods from the ActionScript target.
     */
    public String getTarget64BitStringFromValue(long word) {
        StringBuffer buf = new StringBuffer(22); // enough for the two "0x", "," and " "
        buf.append("0x");
        writeHexWithPadding(buf, Integer.toHexString((int)(word & 0x00000000ffffffffL)));
        buf.append(", 0x");
        writeHexWithPadding(buf, Integer.toHexString((int)(word >> 32)));

        return buf.toString();
    }

    private void writeHexWithPadding(StringBuffer buf, String digits) {
        digits = digits.toUpperCase();
        int padding = 8 - digits.length();
        // pad left with zeros
        for (int i=1; i<=padding; i++) {
            buf.append('0');
        }
        buf.append(digits);
    }
}
