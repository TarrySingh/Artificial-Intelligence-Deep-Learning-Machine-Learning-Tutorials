package org.antlr.runtime {
	import flash.filesystem.File;
	import flash.filesystem.FileMode;
	import flash.filesystem.FileStream;
	import flash.system.System;
	
	public class ANTLRFileStream extends ANTLRStringStream {
		protected var _file:File;
		
		public function ANTLRFileStream(file:File, encoding:String = null) {
			load(file, encoding);
		}

		public function load(file:File, encoding:String = null):void {
			_file = file;
			if (encoding == null) {
				encoding = File.systemCharset;
			}
			
			var stream:FileStream = new FileStream();
			
			try {
				stream.open(file, FileMode.READ);
				data = stream.readMultiByte(file.size, encoding);
				n = data.length;
			}
			finally {
				stream.close();
			}
		}
		
		public override function get sourceName():String {
			return _file.name;
		}
	}
}