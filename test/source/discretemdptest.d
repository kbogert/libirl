import discretemdp;
import tested;
import std.conv;
import std.math;
import mdp : DistInitType;


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

class testObjSpace : Space!(testObj) {

     testObj [] arr;

     public this(int size) {
         for (int i = 0; i < size; i ++)
             arr ~= new testObj(i);
     }

     override public size_t size() {
         return arr.length;
     }

     override public bool contains(testObj i) {
         foreach( a ; arr) 
             if (a == i)
                 return true;
         return false;
     }

     override int opApply(int delegate(ref testObj) dg) {
          int result = 0;
          foreach (value ; arr) {
               result = dg(value);
               if (result) break;

          }
          return result;
     }

}

@name("Distribution Create and foreach")
unittest {

   double[testObj] init;

   init[new testObj()] = 1;

   Distribution!(testObj) dist = new Distribution!(testObj)(init);

   assert(dist.size() == 1, "Distribution size is incorrect");
   assert(dist[new testObj()] == 1.0, "Probability of test object is incorrect");


   int spaceSize = 10;
   testObjSpace tos = new testObjSpace(spaceSize);

   dist = new Distribution!(testObj)(tos);

   assert(dist.size() == 0, "Distribution size is incorrect: " ~ to!string(dist.size()) ~ " should be: " ~ to!string(0));

   assert(dist.toString() == "", "Distribution toString incorrect");

   dist = new Distribution!(testObj)(tos, DistInitType.Uniform);

   assert(dist.size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.size()) ~ " should be: " ~ to!string(spaceSize));

   foreach(key, val ; dist) {
       assert(val == 1.0 / cast(double)spaceSize, "Probability incorrect in uniform distribution");
   }

   debug {
       import std.stdio;
       writeln(dist);
   }

   dist = new Distribution!(testObj)(tos, DistInitType.Exponential, 10.0);

   assert(dist.size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.size()) ~ " should be: " ~ to!string(spaceSize));
   double total = 0;
   foreach(key, val ; dist) {
       total += val;
   }

   debug {
       import std.stdio;
       writeln("Exponential");
       writeln(dist);
   }
   assert(abs(1.0 - total) < TOLERANCE, "Probability distribution not normalized: " ~ to!string(total) ~ " should be 1.0");

   dist = new Distribution!(testObj)(tos, DistInitType.RandomFromGaussian);

   assert(dist.size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.size()) ~ " should be: " ~ to!string(spaceSize));
   total = 0;
   foreach(key, val ; dist) {
       total += val;
   }

   debug {
       import std.stdio;
       writeln();
       writeln(dist);
   }

   assert(abs(1.0 - total) < TOLERANCE, "Probability distribution not normalized: " ~ to!string(total) ~ " should be 1.0");

   
}


@name("Building Distributions with +=")
unittest {
   double[testObj] init;

   init[new testObj()] = 1;

   Distribution!(testObj) dist = new Distribution!(testObj)(init);

   assert(dist.size() == 1, "Distribution size is incorrect");
   assert(dist[new testObj()] == 1.0, "Probability of test object is incorrect");

   testObj a = new testObj();
   a.a = 1;

   dist[a] = 0.0;

   dist.normalize();
   assert(dist.size() == 2, "Distribution size is incorrect");
   assert(dist[a] == 0, "Probability of test object is incorrect: " ~ to!string(dist[a]) ~ " should be: 0");


   for (int i = 2; i < 200; i ++) {
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

   assert(dist[new testObj(199)] == 199.0 / sum, "Normalization Error");
   assert(dist[new testObj(0)] == 1.0 / sum, "Normalization Error");
   assert(dist[new testObj(150)] == 150.0 / sum, "Normalization Error");

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

   assert (dist.argmax() == new testObj(0), "Argmax didn't work!");



}


@name("Distribution iterations")
unittest {

    // generate a distribution and test the iteration methods


    Distribution!(testObj) dist = new Distribution!(testObj)();


    int total = 0;
    
    for (int i = 0; i < 200; i ++) {
        dist[new testObj(i)] = i;
        total += i;
    }

    dist.normalize();


    int counter = 0;
    foreach(key ; dist.byKey()) {
        counter += key.a;
    }

    assert(counter == total, "Didn't iterate through all the keys.");

    double prob = 0;
    
    foreach(val ; dist.byValue()) {
        prob += val;
    }

    assert(abs( prob - 1.0) < TOLERANCE, "Didn't iterate through all the values - " ~ to!string(prob));

    counter = 0;
    prob = 0;

    foreach( T ; dist.byKeyValue()) {
        counter += T.key.a;
        prob += T.value;
    }
    
    assert(counter == total, "Didn't iterate through all the keys.");
    assert(abs( prob - 1.0) < TOLERANCE, "Didn't iterate through all the values - " ~ to!string(prob));

    
    counter = 0;
    prob = 0;

    foreach( key, value ; dist) {
        counter += key.a;
        prob += value;
    }
    
    assert(counter == total, "Didn't iterate through all the keys.");
    assert(abs( prob - 1.0) < TOLERANCE, "Didn't iterate through all the values - " ~ to!string(prob));

    
}

@name("Distribution sampling") 
unittest {

    int distSize = 500;
    int samples = 100000;
    double KLD = 0.0035;

    for (int k = 0; k < 100; k ++) {

        Distribution!(testObj) dist = new Distribution!(testObj)();

        
        for (int i = 0; i < distSize; i ++) {
            dist[new testObj(i)] = 1;
        }

        dist.normalize();

        assert(abs(dist.entropy() - 6.21461) < HALFTOLERANCE, "Distribution entropy is incorrect");

        // create a new distribution from the samples

        Distribution!testObj dist2 = new Distribution!testObj();

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


@name("Empty Distribution") 
unittest {

    Distribution!(testObj) dist = new Distribution!(testObj)();


    try {
        dist.normalize();

        assert(false, "Normalize should throw an exception with an empty distribution");

    } catch (Exception e) {
        // This is supposed to happen
    }

    assert(dist.argmax() is null, "Argmax not working right in an empty distribution");

    foreach(t, v ; dist) {
        assert(false, "There should be nothing to iter over in an empty distribution");
    }

    foreach(key ; dist.byKey()) {
        assert(false, "There should be nothing to iter over in an empty distribution");
    }

    foreach(val ; dist.byValue()) {
        assert(false, "There should be nothing to iter over in an empty distribution");
    }

    foreach( T ; dist.byKeyValue()) {
        assert(false, "There should be nothing to iter over in an empty distribution");
    }

    assert(dist[new testObj(0)] == 0, "Index should have returned a zero for everything in an empty distribution");

    assert(dist.size() == 0, "Size should be zero");

    assert(dist.sample() is null, "Sample should return null");
}


@name("Space tests")
unittest {
    Distribution!(testObj) dist = new Distribution!(testObj)(new testObjSpace(10));

    try {
        dist[new testObj(11)];

        assert(false, "Should not allow access to objects outside of the space");      
    } catch (Exception e) {
        // this is supposed to happen
    }

    
    try {
        dist[new testObj(11)] = 1.0;

        assert(false, "Should not allow access to objects outside of the space");      
    } catch (Exception e) {
        // this is supposed to happen
    }
    
    try {
        dist[new testObj(11)] += 1;

        assert(false, "Should not allow access to objects outside of the space");      
    } catch (Exception e) {
        // this is supposed to happen
    }
}


@name("Distribution optimize")
unittest {

    // generate a distribution and test the iteration methods


    Distribution!(testObj) dist = new Distribution!(testObj)();


    
    for (int i = 0; i < 200; i ++) {
        dist[new testObj(i)] = i;
    }

    dist.normalize();

    assert(dist.size() == 200, "Distribution size is wrong");

    dist.optimize();

    assert(dist.size() == 199, "Distribution size not reduced by optimize()");
    
}



// Function and space experimenting
import std.traits;

class func(RETURN_TYPE, PARAM ...) {

    RETURN_TYPE [PARAM] storage;
    space!PARAM mySpace;

    public this(space!PARAM s) {
        mySpace = s;
    }

    
    RETURN_TYPE opIndex(PARAM i) {
        RETURN_TYPE* p;
        p = (i in storage);
        if (p !is null) {
            return *p;
        }
        if ( mySpace !is null && ! mySpace.contains(i)) {
            throw new Exception("ERROR, key is not in the space this function is defined over.");
        }
        return 0;
    }


    void opIndexAssign(RETURN_TYPE value, PARAM i) {
          if ( mySpace !is null && ! mySpace.contains(i)) {
               throw new Exception("ERROR, key is not in the space this function is defined over.");
          }
          storage[i] = value;
    }


    // FOR NUMERIC RETURN TYPES ONLY
    void opIndexOpAssign(string op)(RETURN_TYPE rhs, T key)
        if ( isNumeric!(RETURN_TYPE))
    {
        RETURN_TYPE* p;
        p = (key in storage);
        if (p is null) {
            if ( mySpace !is null && ! mySpace.contains(key)) {
                throw new Exception("ERROR, key is not in the space this distribution is defined over.");
            }
            storage[key] = 0;
            p = (key in myDistribution);
        }
        mixin("*p " ~ op ~ "= rhs;");
    }    
    
    auto byKey() {
        return storage.byKey();
    }

    auto byValue() {
        return storage.byValue();
    }

    auto byKeyValue() {
        return storage.byKeyValue();
    }


    RETURN_TYPE max() 
        if (PARAM.length == 1)
    {
       
        RETURN_TYPE max;
        bool setMax = false;
        
        foreach (auto key : mySpace) {

            RETURN_TYPE val = storage[key];
            
            if (! setMax ) {
                max = val;
                setMax = true;
            } else {
                if (val > max) {
                    max = val;
                }
            }                
            
        }

        return max;

    }

            
    func!(RETURN_TYPE, PARAM[0 .. PARAM.length - 2] ) max() 
        if (PARAM.length > 1)
    {
        alias SUBPARAM PARAM[0 .. PARAM.length - 2]

        auto newSpace = mySpace.orth_project!(SUBPARAM)(true);
        
        auto returnval = new func!(RETURN_TYPE,SUBPARAM)(newSpace);

        foreach (auto key : newSpace) {

            RETURN_TYPE max;
            bool setMax = false;

            foreach( auto subkey : mySpace.orth_project!(PARAM[PARAM.length - 1])(false) ) {

                auto combinedKey = Tuple!( key , subkey );  // THIS MAYBE COULD BE AVOIDED WITH MIXINS, DEFINING THE ASSOCIATIVE ARRAY PER DIMENSION, AND THIS ALLOWS FOR PARTIAL ADDRESSING
                RETURN_TYPE val = storage[ combinedKey ];           // BUT, ONLY IF THE LAST DIMENSION IS THE ONE MAXXED OVER, IF MORE THAN ONE THIS WOULDN'T WORK
                
                if (! setMax ) {
                    max = val;
                    setMax = true;
                } else {
                    if (val > max) {
                        max = val;
                    }
                }                
                
            }

            returnval[key] = max;   
        }

        return returnval;

    }

        
}





import std.typetuple;

class space(T ...) {

    abstract public size_t size();
    abstract public bool contains(T i);
    abstract int opApply(int delegate(ref T) dg);
    abstract space(PROJECTED_DIMS) orth_project(PROJECTED_DIMS)(bool frontDimsFirst = true)
        if (PROJECTED_DIMS.length > 0 && allSatisfy!(dimOfSpace, PROJECTED_DIMS)) ;


    protected template dimOfSpace(DIM) {
        enum dimOfSpace = (staticIndexOf!(DIM, T) != -1);
    }
}


// now create some spaces, 1D, 2D, 3D


