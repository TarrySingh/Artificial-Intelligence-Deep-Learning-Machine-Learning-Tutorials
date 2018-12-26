/** 
 * Loads the contents of a file all at once and passes its contents off to
 * {@link org.antlr.runtime.ANTLRStringStream}.
 * Currently this class can only be used in the Rhino JS interpreter.
 * @class
 * @extends org.antlr.runtime.ANTLRStringStream
 * @param {String} fileName path of the file to be loaded
 * @param {String} [encoding] name of the charset used for decoding
 */
org.antlr.runtime.ANTLRFileStream = function(fileName, encoding) {
    this.fileName = fileName;

    // @todo need to add support for other JS interpreters that have file i/o
    // hooks (SpiderMonkey and WSH come to mind).
    var method;
    if (org.antlr.env.ua.rhino) {
        method = "loadFileUsingJava";
    } else {
        throw new Error(
            "ANTLR File I/O is not supported in this JS implementation."
        );
    }

    var data = this[method](fileName, encoding);
    org.antlr.runtime.ANTLRFileStream.superclass.constructor.call(this, data);
};

org.antlr.lang.extend(org.antlr.runtime.ANTLRFileStream,
                  org.antlr.runtime.ANTLRStringStream,
/** @lends org.antlr.runtime.ANTLRFileStream.prototype */{
    /**
     * Get the file path from which the input was loaded.
     * @returns {String} the file path from which the input was loaded
     */
    getSourceName: function() {
        return this.fileName;
    },

    /**
     * Read the file and return its contents as a JS string.
     * @private
     * @param {String} fileName path of the file to be loaded
     * @param {String} [encoding] name of the charset used for decoding
     * @returns {String} the contents of the file
     */
    loadFileUsingJava: function(fileName, encoding) {
        // read the file using Java methods
        var f = new java.io.File(fileName),
            size = f.length(),
            isr,
            fis = new java.io.FileInputStream(f);
        if (encoding) {
            isr = new java.io.InputStreamReader(fis, encoding);
        } else {
            isr = new java.io.InputStreamReader(fis);
        }
        var data = java.lang.reflect.Array.newInstance(java.lang.Character.TYPE, size);
        isr.read(data, 0, size);

        // convert java char array to a javascript string
        return new String(new java.lang.String(data));
    }
});
