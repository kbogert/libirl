import discretefunctions;
import tested;
import std.conv;
import std.math;
import std.typecons;



double TOLERANCE = 0.000000001;
double HALFTOLERANCE = 0.00001;
class testObj {

   int a;

    public this() {
        a = 0;
    }
    public this(int b) {
        a = b;
    }

    override bool opEquals(Object o) {
          auto rhs = cast(testObj)o;
          if (!rhs) return false;

          return a == rhs.a;
     }

     override size_t toHash() @trusted nothrow {
          return a;
     }

     override string toString() {
          return to!string(a);
     }   
}


class testObj2 {

    int a;

    public this() {
        a = 0;
    }
    public this(int b) {
        a = b;
    }

    override bool opEquals(Object o) {
          auto rhs = cast(testObj2)o;
          if (!rhs) return false;

          return a == rhs.a;
     }

     override size_t toHash() @trusted nothrow {
          return a;
     }

     override string toString() {
          return to!string(a);
     }   
}


@name("Distribution Create and foreach")
unittest {

   double[Tuple!(testObj)] init;

   init[tuple(new testObj())] = 1;
   
   testObjSet testSet1 = new testObjSet(1);

   distribution!(testObj) dist = new distribution!(testObj)(testSet1, init);

   assert(dist.param_set().size() == 1, "Distribution size is incorrect");
   assert(dist[tuple(new testObj())] == 1.0, "Probability of test object is incorrect");

   int spaceSize = 10;

   testSet1 = new testObjSet(spaceSize);
   

   dist = new distribution!(testObj)(testSet1);

   assert(dist.param_set().size() == 10, "Distribution size is incorrect: " ~ to!string(dist.param_set().size()) ~ " should be: " ~ to!string(10));

   assert(dist.toString() != "", "Distribution toString incorrect");

    
   dist = new distribution!(testObj)(testSet1, DistInitType.Uniform);

   assert(dist.param_set().size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.param_set().size()) ~ " should be: " ~ to!string(spaceSize));

   foreach(key ; dist.param_set()) {
       auto val = dist[key];
       assert(val == 1.0 / cast(double)spaceSize, "Probability incorrect in uniform distribution");
   }

   debug {
       import std.stdio;
       writeln(dist);
   }

   dist = new distribution!(testObj)(testSet1, DistInitType.Exponential, 10.0);

   assert(dist.param_set().size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.param_set().size()) ~ " should be: " ~ to!string(spaceSize));
   double total = 0;
   foreach(key; dist.param_set()) {
       auto val = dist[key];
       total += val;
   }

   debug {
       import std.stdio;
       writeln("Exponential");
       writeln(dist);
   }
   assert(abs(1.0 - total) < TOLERANCE, "Probability distribution not normalized: " ~ to!string(total) ~ " should be 1.0");

   dist = new distribution!(testObj)(testSet1, DistInitType.RandomFromGaussian);

   assert(dist.param_set().size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.param_set().size()) ~ " should be: " ~ to!string(spaceSize));
   total = 0;
   foreach(key; dist.param_set()) {
       auto val = dist[key];
       total += val;
   }

   debug {
       import std.stdio;
       writeln();
       writeln(dist);
   }

   assert(abs(1.0 - total) < TOLERANCE, "Probability distribution not normalized: " ~ to!string(total) ~ " should be 1.0");

   dist.optimize();
}


@name("Building Distributions with +=")
unittest {
   double[Tuple!(testObj)] init;

   init[tuple(new testObj())] = 1;

   testObjSet testSet1 = new testObjSet(1);

   distribution!(testObj) dist = new distribution!(testObj)(testSet1, init);

   assert(dist.param_set().size() == 1, "Distribution size is incorrect");
   assert(dist[new testObj()] == 1.0, "Probability of test object is incorrect");

   testObj a = new testObj();
   a.a = 1;

   testSet1 = new testObjSet(2);
   dist = new distribution!(testObj)(testSet1, init);

   dist[a] = 0.0;

   dist.normalize();
   assert(dist.param_set().size() == 2, "Distribution size is incorrect");
   assert(dist[a] == 0, "Probability of test object is incorrect: " ~ to!string(dist[a]) ~ " should be: 0");

   testSet1 = new testObjSet(200);
   dist = new distribution!(testObj)(testSet1);
   
   for (int i = 0; i < 200; i ++) {
       dist[new testObj(i)] += i;
   }

   assert(! dist.isNormalized(), "Distribution should not be normalized");

   dist.normalize();

 /*  debug {
       import std.stdio;
       writeln();
       writeln(dist);
   }*/

   int sum = 0;
   foreach(b ; 0 .. 200)
       sum += b;

   assert(dist[new testObj(199)] == 199.0 / sum, "Normalization Error on item 199");
   assert(dist[new testObj(0)] == 0.0 / sum, "Normalization Error on item 0");
   assert(dist[new testObj(150)] == 150.0 / sum, "Normalization Error on item 150");

   for (int i = 0; i < 200; i ++) {
       dist[new testObj(199 - i)] += i;
   }

   assert(! dist.isNormalized(), "Distribution should not be normalized");

/*   debug {
       import std.stdio;
       writeln();
       writeln(dist);
   }*/


   dist.normalize();

   assert (dist.argmax()[0] == new testObj(0), "Argmax didn't work!");



}

version(fullunittest) {

@name("Distribution sampling") 
unittest {

    int distSize = 500;
    int samples = 100000;

    double KLD = 0.0035;

    testObjSet testSet1 = new testObjSet(distSize);

    for (int k = 0; k < 100; k ++) {

        distribution!(testObj) dist = new distribution!(testObj)(testSet1, DistInitType.Uniform);

        assert(abs(dist.entropy() - 6.21461) < HALFTOLERANCE, "Distribution entropy is incorrect");

        // create a new distribution from the samples

        distribution!testObj dist2 = new distribution!testObj(testSet1);

        for (int j = 0; j < samples; j ++) {
            dist2[dist.sample()] += 1;
        }

        dist2.normalize();

        // compare entropy of the two distributions, should about match
        double theKld = dist.KLD(dist2);

        assert(theKld > 0, "KLD not working, it is <= 0");        
        assert(theKld < KLD, "Sampled distribution is too different from primary: " ~ to!string(theKld) ~ " > " ~ to!string(KLD));

        assert(abs(dist.crossEntropy(dist2) - 6.21461) <= theKld , "Cross entropy incorrect" );
    }


    
}

}


@name("Empty Distribution") 
unittest {

    testObjSet testSet1 = new testObjSet(0);
    
    distribution!(testObj) dist = new distribution!(testObj)(testSet1);


    try {
        dist.normalize();

        assert(false, "Normalize should throw an exception with an empty distribution");

    } catch (Exception e) {
        // This is supposed to happen
    }

    assert(dist.argmax()[0] is null, "Argmax not working right in an empty distribution");

    foreach(key ; dist.param_set()) {
        assert(false, "There should be nothing to iter over in an empty distribution");
    }

    try {
        dist[new testObj(0)] = 0;
        
        assert(false, "Index should have cause an exception in an empty distribution");

    } catch (Exception e) {
        // this is supposed to happen
    }
    
    assert(dist.param_set().size() == 0, "Size should be zero");

    try {
        dist.sample();
        
        assert(false, "Sample should throw an exception in an empty distribution");

    } catch (Exception e) {
        // supposed to happen
    }
}


@name("Multidimensional distributions") 
unittest{

    // test a two dimensional distribution, State action state for example
    // make sure indexing works correctly as well as normalize

    int dimSize = 5;

    testObjSet testSet1 = new testObjSet(dimSize);

    auto fullSet = testSet1.cartesian_product(testSet1);

    auto dist = new distribution!(testObj, testObj)(fullSet, DistInitType.Uniform);

    foreach(key; fullSet) {
        assert(dist[key] == 1.0 / (dimSize * dimSize), "Distribution is not uniform");
    }

    auto one = new testObj(1);
    auto two = new testObj(2);

    assert(dist[one, two] == 1.0 / (dimSize * dimSize), "Indexing not working right");
    assert(dist[tuple(one, two)] == 1.0 / (dimSize * dimSize), "Indexing not working right");

    
}





class testObjSet : set!(testObj) {

    public this(int size) {
        Tuple!(testObj) [] tempArr;
        for (int i = 0; i < size; i ++)
            tempArr ~= tuple(new testObj(i));

        super(tempArr);
    }

    public this(set!(testObj) toCopy) {
        super(toCopy.storage.dup);
    }
}


class testObj2Set : set!(testObj2) {

    public this(int size) {
        Tuple!(testObj2) [] tempArr;
        for (int i = 0; i < size; i ++)
            tempArr ~= tuple(new testObj2(i));

        super(tempArr);
    }

    public this(set!(testObj2) toCopy) {
        super(toCopy.storage.dup);
    }
}

import std.stdio;

@name("Set Create and foreach")
unittest {

    int size = 10;
    
    
    testObjSet testSet = new testObjSet(size);

    assert(testSet.size() == size, "Set size is incorrect");

    int counter = 0;
    int sum = 0;
    foreach( obj ; testSet) {
        sum += obj[0].a;
        counter ++;
    }
    assert (counter == size, "Set foreach did not go over all entries");
    assert (sum == 45, "Set foreach did not go over all entries");
    
}


@name("Set Cartesian product")
unittest {

    int size = 10;
    
    
    testObjSet testSet = new testObjSet(size);


    set!(testObj, testObj) newSet = testSet.cartesian_product(testSet);
    
    assert(newSet.size() == size * size, "Set size is incorrect");

    foreach (obj ; newSet) {
        assert(newSet.contains(obj));
    }

    set!(testObj, testObj, testObj) bigSet = newSet.cartesian_product(testSet);

    assert(bigSet.size() == size * size * size, "Set size is incorrect");
    
}

@name("Set projection")
unittest {

    int size = 10;
    
    
    testObjSet testSet = new testObjSet(size);


    set!(testObj, testObj) newSet = testSet.cartesian_product(testSet);

    set!(testObj, testObj, testObj) bigSet = newSet.cartesian_product(testSet);

    testObjSet finalSet = new testObjSet( newSet.orth_project!(testObj)() );


    assert(finalSet.size() == size, "Set size is incorrect");

    int counter = 0;
    int sum = 0;
    foreach( obj ; finalSet) {
        sum += obj[0].a;
        counter ++;
    }
    assert (counter == size, "Set foreach did not go over all entries");
    assert (sum == 45, "Set foreach did not go over all entries");

    finalSet = new testObjSet( bigSet.orth_project!(testObj)() );

    assert(finalSet.size() == size, "Set size is incorrect");

    newSet = bigSet.orth_project!(testObj, testObj)();

    assert(newSet.size() == size * size, "Set size is incorrect");

    set!(testObj, testObj) newSet2 = newSet.orth_project!(testObj, testObj)();

}

@name("Correct Dimension Removed")
unittest {

    int size = 10;
    int size2 = 5;


    testObjSet testSet1 = new testObjSet(size);
    testObj2Set testSet2 = new testObj2Set(size2);
    

    set!(testObj, testObj2, testObj) bigset = testSet1.cartesian_product(testSet2).cartesian_product(testSet1);

    
    // back
    
    set!(testObj, testObj2) attempt1 = bigset.orth_project!(testObj, testObj2)();

    assert(attempt1.size() == size * size2, "Set size is incorrect");
    
    attempt1 = bigset.remove_dim_back!(testObj)();    

    assert(attempt1.size() == size * size2, "Set size is incorrect");
    
    set!(testObj, testObj) attempt2  = bigset.remove_dim_back!(testObj2)();

    assert(attempt2.size() == size * size, "Set size is incorrect");

    set!(testObj) attempt3 = bigset.remove_dim_back!(testObj2, testObj)();
    

    // front

       
    set!(testObj2, testObj) attempt4 = bigset.remove_dim_front!(testObj)();    

    assert(attempt4.size() == size * size2, "Set size is incorrect");
    
    set!(testObj, testObj) attempt5  = bigset.remove_dim_front!(testObj2)();

    assert(attempt5.size() == size * size, "Set size is incorrect");

    set!(testObj) attempt6 = bigset.remove_dim_front!(testObj, testObj2)();

    // should not work
 //   attempt6 = bigset.remove_dim_front!(testObj2, testObj)();
 //   attempt6 = bigset.remove_dim_back!(testObj, testObj2)();
        
        
}



@name("Create function")
unittest {
    int size = 10;


    testObjSet testSet1 = new testObjSet(size);

    func!(double, testObj) testFunc = new func!(double, testObj)(testSet1, 0.0);

    set!(testObj, testObj) testSet2 = testSet1.cartesian_product(testSet1);

    func!(double, testObj, testObj) testFunc2 = new func!(double, testObj, testObj)(testSet2, 0.0);

    foreach (key ; testSet1) {

        testFunc[key] = key[0].a;
    }


    foreach (key ; testSet2) {

        testFunc2[key] = key[0].a + key[1].a;
    }
}

@name("Function max and argmax")
unittest {
    int size = 10;


    testObjSet testSet1 = new testObjSet(size);

    func!(double, testObj) testFunc = new func!(double, testObj)(testSet1, 0.0);

    set!(testObj, testObj) testSet2 = testSet1.cartesian_product(testSet1);

    func!(double, testObj, testObj) testFunc2 = new func!(double, testObj, testObj)(testSet2, 0.0);

    foreach (key ; testSet1) {

        testFunc[key] = key[0].a;
    }


    foreach (key ; testSet2) {

        testFunc2[key] = key[0].a + key[1].a;
    }
    assert(testFunc.max() == 9, "Max did not work");

    assert(testFunc2.max().max() == 18, "Max did not work");

    func!(double, testObj) max1 = testFunc2.max!(testObj)();

    foreach (key; max1.param_set()) {
        assert(max1[key] == key[0].a + 9, "Something is wrong with the max calculation");
    }


    func!(Tuple!(testObj), testObj) testArgMax = testFunc2.argmax();

    foreach (key; testArgMax.param_set()) {

        assert(testArgMax[key][0].a == 9, "ArgMax didn't work right");
    }
    
    testArgMax = testFunc2.argmax!(testObj)();

    foreach (key; testArgMax.param_set()) {

        assert(testArgMax[key][0].a == 9, "ArgMax didn't work right");
    }

    assert(testFunc2.argmax().argmax()[0].a >= 0 , "Argmax didn't work right");
       
}


@name("Function sumout")
unittest {
    int size = 10;


    testObjSet testSet1 = new testObjSet(size);

    func!(double, testObj) testFunc = new func!(double, testObj)(testSet1, 0.0);

    set!(testObj, testObj) testSet2 = testSet1.cartesian_product(testSet1);

    func!(double, testObj, testObj) testFunc2 = new func!(double, testObj, testObj)(testSet2, 0.0);

    foreach (key ; testSet1) {

        testFunc[key] = key[0].a;
    }


    foreach (key ; testSet2) {

        testFunc2[key] = key[1].a;
    }
    assert(testFunc.sumout() == 45, "Sumout did not work");

    auto sum = testFunc2.sumout();

    foreach (key ; sum.param_set()) {
        assert( sum[key] == 45, "Sumout not correct for each element" );
    }    
    assert(testFunc2.sumout().sumout() == 450, "Sumout did not work 2");
}


@name("Function apply")
unittest {
    int size = 10;


    testObjSet testSet1 = new testObjSet(size);

    func!(double, testObj) testFunc = new func!(double, testObj)(testSet1, 0.0);

    set!(testObj, testObj) testSet2 = testSet1.cartesian_product(testSet1);

    func!(double, testObj, testObj) testFunc2 = new func!(double, testObj, testObj)(testSet2, 0.0);

    foreach (key ; testSet1) {

        testFunc[key] = key[0].a;
    }

    foreach (key ; testSet2) {

        testFunc2[key] = key[1].a;
    }

    func!(Tuple!(testObj), testObj) testArgMax = testFunc2.argmax();

    func!(double, testObj) testFunc3 = testFunc2.apply(testArgMax);

    foreach(key ; testFunc3.param_set) {
        assert(testFunc3[key] == 9, "Apply did not select correct items");
    }

    
}

@name("Function ops")
unittest {

    int size = 10;


    testObjSet testSet1 = new testObjSet(size);

    func!(double, testObj) testFunc = new func!(double, testObj)(testSet1, 0.0);

        foreach (key ; testSet1) {

        testFunc[key] = key[0].a;
    }

    func!(double, testObj) result = testFunc + testFunc;

    foreach(key ; result.param_set) {
        assert(result[key] == key[0].a + key[0].a, "Sum did not work right");
    }
    
    result = testFunc * testFunc;

    foreach(key ; result.param_set) {
        assert(result[key] == key[0].a * key[0].a, "mult did not work right");
    }

    result = testFunc - testFunc;

    foreach(key ; result.param_set) {
        assert(result[key] == key[0].a - key[0].a, "Subtract did not work right");
    }


    result = testFunc / testFunc;
    
    foreach(key ; result.param_set) {
        if (! (key[0].a == 0 && key[0].a == 0))
            assert(result[key] == cast(double)key[0].a / cast(double)key[0].a, "Divide did not work right " ~ to!string(key[0].a) ~ " / " ~ to!string(key[0].a) ~ " = " ~ to!string(cast(double)key[0].a / cast(double)key[0].a) ~ " != " ~ to!string(result[key]));
    }


    
    set!(testObj, testObj) testSet2 = testSet1.cartesian_product(testSet1);

    func!(double, testObj, testObj) testFunc2 = new func!(double, testObj, testObj)(testSet2, 0.0);
    
    foreach (key ; testSet2) {

        testFunc2[key] = key[1].a;
    }


    func!(double, testObj, testObj) result2 = testFunc2 + testFunc;

    foreach (key ; testSet2) {

        assert(result2[key] == key[1].a + key[1].a, "Assymetrical sum did not work right");
    }

        
    result2 = testFunc2 + testFunc2;

    foreach (key ; testSet2) {

        assert(result2[key] == key[1].a + key[1].a, "Symetrical sum did not work right");
    }    


    // tests opBinary
    result2 = testFunc2 * 0.5;
    
    foreach (key ; testSet2) {

        assert(result2[key] == key[1].a * 0.5, "Scalar multiply did not work right");
    }    

    // tests opBinaryRight
    result2 = 0.5 * testFunc2;
    
    foreach (key ; testSet2) {

        assert(result2[key] == key[1].a * 0.5, "Scalar multiply did not work right");
    }    
    
}
    
