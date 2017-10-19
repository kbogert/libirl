module app;

import tested;


void main()
{
	version (unittest) {} else {
		import std.stdio;
		writeln(`This application does nothing. Run with "dub --build=unittest"`);
	}
}

shared static this()
{
	version (unittest) {
		// disable built-in unit test runner
		import gridworld;
		import core.runtime;
		Runtime.moduleUnitTester = () => true;
		assert(runUnitTests!gridworld(new JsonTestResultWriter("results.json")), "Unit tests failed.");
	}
}
