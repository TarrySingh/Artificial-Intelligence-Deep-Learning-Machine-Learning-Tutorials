"""This is a substantially improved version of the older Interpreter.py demo
It creates a simple GUI JPython console window with simple history
as well as the ability to interupt running code (with the ESC key).

Like Interpreter.py, this is still just a demo, and needs substantial
work before serious use.

Thanks to Geza Groma (groma@everx.szbk.u-szeged.hu) for several valuable
ideas for this tool -- his JPConsole is a more refined implementation
of similar ideas.
"""

from Styles import Styles
from Keymap import Keymap

from pawt import swing, colors
from java.awt.event.KeyEvent import VK_UP, VK_DOWN
from java.awt.event import ActionEvent
from java.lang import Thread, System
from code import compile_command
import string, sys, re

class OutputBuffer:
	def __init__(self, console, stylename):
		self.console = console
		self.stylename = stylename
		
	def flush(self):
		pass
		
	def write(self, text):
		self.console.write(text, self.stylename)

class Console:
	def __init__(self, styles=None, keymap=None):
		if styles is None:
			styles = Styles()
			basic = styles.add('normal', tabsize=3, fontSize=12, fontFamily="Courier")
			styles.add('error', parent=basic, foreground=colors.red)
			styles.add('output', parent=basic, foreground=colors.blue)
			styles.add('input', parent=basic, foreground=colors.black)
			styles.add('prompt', parent=basic, foreground=colors.purple)
		self.styles = styles
		
		# This is a hack to get at an inner class
		# This will not be required in JPython-1.1
		ForegroundAction = getattr(swing.text, 'StyledEditorKit$ForegroundAction')
		self.inputAction = ForegroundAction("start input", colors.black)

		if keymap is None:
			keymap = Keymap()
		keymap.bind('enter', self.enter)
		keymap.bind('tab', self.tab)
		keymap.bind('escape', self.escape)
		keymap.bind('up', self.uphistory)
		keymap.bind('down', self.downhistory)
		
		self.keymap = keymap
		
		self.document = swing.text.DefaultStyledDocument(self.styles)
		self.document.setLogicalStyle(0, self.styles.get('normal'))

		self.textpane = swing.JTextPane(self.document)
		self.textpane.keymap = self.keymap
		
		self.history = []
		self.oldHistoryLength = 0
		self.historyPosition = 0
		
		self.command = []
		self.locals = {}

	def write(self, text, stylename='normal'):
		style = self.styles.get(stylename)
		self.document.insertString(self.document.length, text, style)
		
	def beep(self):
		self.textpane.toolkit.beep()

	def startUserInput(self, prompt=None):
		if prompt is not None:
			self.write(prompt, 'prompt')
		self.startInput = self.document.createPosition(self.document.length-1)
		#self.document.setCharacterAttributes(self.document.length-1, 1, self.styles.get('input'), 1)
		self.textpane.caretPosition = self.document.length
		ae = ActionEvent(self.textpane, ActionEvent.ACTION_PERFORMED, 'start input')
		self.inputAction.actionPerformed(ae)

	def getinput(self):
		offset = self.startInput.offset
		line = self.document.getText(offset+1, self.document.length-offset)
		return string.rstrip(line)

	def replaceinput(self, text):
		offset = self.startInput.offset + 1
		self.document.remove(offset, self.document.length-offset)
		self.write(text, 'input')
		
	def enter(self):
		line = self.getinput()
		self.write('\n', 'input')
		
		self.history.append(line)
		self.handleLine(line)
		
	def gethistory(self, direction):
		historyLength = len(self.history)
		if self.oldHistoryLength < historyLength:
			# new line was entered after last call
			self.oldHistoryLength = historyLength
			if self.history[self.historyPosition] != self.history[-1]:
				self.historyPosition = historyLength

		pos = self.historyPosition + direction

		if 0 <= pos < historyLength:
			self.historyPosition = pos
			self.replaceinput(self.history[pos])
		else:
			self.beep()

	def uphistory(self):
		self.gethistory(-1)

	def downhistory(self):
		self.gethistory(1)

	def tab(self):
		self.write('\t', 'input')
		
	def escape(self):
		if (not hasattr(self, 'pythonThread') or self.pythonThread is None or not self.pythonThread.alive):
			self.beep()
			return
			
		self.pythonThread.stopPython()

	def capturePythonOutput(self, stdoutStyle='output', stderrStyle='error'):
		import sys
		sys.stdout = OutputBuffer(self, stdoutStyle)
		sys.stderr = OutputBuffer(self, stderrStyle)

	def handleLine(self, text):
		self.command.append(text)
		
		try:
			code = compile_command(string.join(self.command, '\n'))
		except SyntaxError:
			traceback.print_exc(0)
			self.command = []
			self.startUserInput(str(sys.ps1)+'\t')
			return

		if code is None:
			self.startUserInput(str(sys.ps2)+'\t')
			return
		
		self.command = []
		
		pt = PythonThread(code, self)
		self.pythonThread = pt
		pt.start()
		
	def newInput(self):
		self.startUserInput(str(sys.ps1)+'\t')
		
import traceback

class PythonThread(Thread):
	def __init__(self, code, console):
		self.code = code
		self.console = console
		self.locals = console.locals
		
	def run(self):
		try:
			exec self.code in self.locals
			
		#Include these lines to actually exit on a sys.exit() call
		#except SystemExit, value:
		#	raise SystemExit, value
		
		except:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			l = len(traceback.extract_tb(sys.exc_traceback))
			try:
				1/0
			except:
				m = len(traceback.extract_tb(sys.exc_traceback))
			traceback.print_exception(exc_type, exc_value, exc_traceback, l-m)
			
		self.console.newInput()

	def stopPython(self):
		#Should spend 2 seconds trying to kill thread in nice Python style first...
		self.stop()

header = """\
JPython %(version)s on %(platform)s
%(copyright)s
""" % {'version':sys.version, 'platform':sys.platform, 'copyright':sys.copyright}

if __name__ == '__main__':
	c = Console()
	pane = swing.JScrollPane(c.textpane)
	swing.test(pane, size=(500,400), name='JPython Console')
	c.write(header, 'output')
	c.capturePythonOutput()
	c.textpane.requestFocus()
	c.newInput()
