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
		import core.runtime;
		Runtime.moduleUnitTester = () => true;
        
        

        version(fullunittest) {
            import gridworldmdptest;
            import gridworldirltest;
            import discretefunctionstest;
            import randommdptest;
            import trajectorytest;
            import occlusiontest;
            import solvers;
            
    //		runUnitTests!(gridworld)(new JsonTestResultWriter("results.json"));

            bool allSuccessful = true;
        
    		allSuccessful &= runUnitTests!(solverstest)(new ConsoleTestResultWriter);
    		allSuccessful &= runUnitTests!(occlusiontest)(new ConsoleTestResultWriter);
    		allSuccessful &= runUnitTests!(trajectorytest)(new ConsoleTestResultWriter);
    		allSuccessful &= runUnitTests!(randommdptest)(new ConsoleTestResultWriter);
            allSuccessful &= runUnitTests!(gridworldmdptest)(new ConsoleTestResultWriter);
    		allSuccessful &= runUnitTests!(gridworldirltest)(new ConsoleTestResultWriter);
    		allSuccessful &= runUnitTests!(discretefunctionstest)(new ConsoleTestResultWriter);

            assert(allSuccessful, "Unit tests failed.");
        } else {
            import singletest;

            assert(runUnitTests!(singletest)(new ConsoleTestResultWriter));

        }
	}
}
