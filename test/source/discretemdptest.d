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

version(full) {

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



// Function and set experimenting
import std.traits;

class func(RETURN_TYPE, PARAM ...) {

    RETURN_TYPE [PARAM] storage;
    set!PARAM mySet;

    RETURN_TYPE funct_default;

    public this(set!PARAM s, RETURN_TYPE def) {
        mySet = s;
        funct_default = def;
    }

    
    RETURN_TYPE opIndex(PARAM i) {
        RETURN_TYPE* p;
        p = (i in storage);
        if (p !is null) {
            return *p;
        }
        if ( mySet !is null && ! mySet.contains(i)) {
            throw new Exception("ERROR, key is not in the set this function is defined over.");
        }
        return funct_default;
    }


    void opIndexAssign(RETURN_TYPE value, PARAM i) {
          if ( mySet !is null && ! mySet.contains(i)) {
               throw new Exception("ERROR, key is not in the set this function is defined over.");
          }
          storage[i] = value;
    }


    // FOR NUMERIC RETURN TYPES ONLY
    void opIndexOpAssign(string op)(RETURN_TYPE rhs, T key) {
        RETURN_TYPE* p;
        p = (key in storage);
        if (p is null) {
            if ( mySet !is null && ! mySet.contains(key)) {
                throw new Exception("ERROR, key is not in the set this distribution is defined over.");
            }
            storage[key] = funct_def;
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


    static if (PARAM.length == 1) {
        

        RETURN_TYPE max() {
       
            RETURN_TYPE max;
            bool setMax = false;
        
            foreach (key ; mySet) {

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

        Tuple!(PARAM) argmax() {
            RETURN_TYPE max;
            Tuple!(PARAM) max_param;
            bool setMax = false;
        
            foreach (key ; mySet) {

                RETURN_TYPE val = storage[key];
            
                if (! setMax ) {
                    max = val;
                    max_param = key;
                    setMax = true;
                } else {
                    if (val > max) {
                        max = val;
                        max_param = key;
                    }
                }                
            
            }

            return max_param;
        }
        
    } else {
        
        func!(RETURN_TYPE, PARAM[0 .. PARAM.length - 2] ) max() {

            return max(PARAM[PARAM.length - 1])();

        }


        func!(RETURN_TYPE, removeLast(TOREMOVE) ) max(TOREMOVE...)() 
            if (TOREMOVE.length > 0 && allSatisfy!(dimOfSet, TOREMOVE))
        {
            alias SUBPARAM = removeLast!(TOREMOVE);

            auto newSet = mySet.orth_project!(SUBPARAM)();
        
            auto returnval = new func!(RETURN_TYPE,SUBPARAM)(newSet);

            foreach (key ; newSet) {

                RETURN_TYPE max;
                bool setMax = false;

                foreach( subkey ; mySet.remove_dim_front!(SUBPARAM)() ) {

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

        func!(Tuple!(PARAM[PARAM.length - 1]), PARAM[0 .. PARAM.length - 2] ) argmax() {

            return argmax(PARAM[PARAM.length - 1])();

        }
        
        func!(Tuple!(TOREMOVE), removeLast(TOREMOVE) ) argmax(TOREMOVE...)() 
            if (TOREMOVE.length > 0 && allSatisfy!(dimOfSet, TOREMOVE))

        {
            alias SUBPARAM = removeLast!(TOREMOVE);

            auto newSet = mySet.orth_project!(SUBPARAM)();
        
            auto returnval = new func!(Tuple!(TOREMOVE),SUBPARAM)(newSet);

            foreach (key ; newSet) {

                RETURN_TYPE max;
                Tuple!(TOREMOVE) max_key;
                bool setMax = false;

                foreach( subkey ; mySet.remove_dim_front!(SUBPARAM)() ) {

                    auto combinedKey = Tuple!( key , subkey );  // THIS MAYBE COULD BE AVOIDED WITH MIXINS, DEFINING THE ASSOCIATIVE ARRAY PER DIMENSION, AND THIS ALLOWS FOR PARTIAL ADDRESSING
                    RETURN_TYPE val = storage[ combinedKey ];           // BUT, ONLY IF THE LAST DIMENSION IS THE ONE MAXXED OVER, IF MORE THAN ONE THIS WOULDN'T WORK
                
                    if (! setMax ) {
                        max = val;
                        max_key = subkey;
                        setMax = true;
                    } else {
                        if (val > max) {
                            max = val;
                            max_key = subkey;
                        }
                    }                
                
                }

                returnval[key] = max_key;   
            }

            return returnval;
        }
        
    }


    
    protected template removeLast(FIRST, T ...) {
        static if (T.length > 0) {
            auto removeLast = Reverse!(Erase!(FIRST, Reverse!(  removeLast(T) )));
        } else {
            auto removeLast = Reverse!(Erase!(FIRST, Reverse!(PARAM)));
        }
    }

    
    protected template dimOfSet(DIM) {
        enum dimOfSet = (staticIndexOf!(DIM, PARAM) != -1);
    }    
}





import std.typetuple;
import std.typecons;

// now create some sets, 1D, 2D, 3D


// could be optimized for 1D sets by removing the tuples, if I feel like it I guess
class set(T ...) {

    Tuple!(T) [] storage;


    public this(Tuple!(T) [] elements) {
        storage = elements;
    }
    
    public size_t size() {
        return storage.length;
    }
    
    public bool contains(Tuple!(T) i) {
         foreach( a ; storage) 
             if (a == i)
                 return true;
         return false;
    }

    int opApply(int delegate(ref Tuple!(T)) dg) {
          int result = 0;
          foreach (value ; storage) {
               result = dg(value);
               if (result) break;

          }
          return result;
    }
    
    set!(PROJECTED_DIMS) orth_project(PROJECTED_DIMS...)()
        if (PROJECTED_DIMS.length > 0 && allSatisfy!(dimOfSet, PROJECTED_DIMS)) 
    {
        return remove_dim_back!( removeFirst!(PROJECTED_DIMS) )();
    }

    set!( removeFirst!(DIMS) ) remove_dim_front(DIMS...)()
        if (DIMS.length > 0 && allSatisfy!(dimOfSet, DIMS)) 
    {

        alias NEWDIMS = removeFirst!(DIMS);
        
        bool [Tuple!(NEWDIMS)] newElements;

        // go through storage, only adding unique tuples to newElements
        // this is harder than it needs to be, really would be easy with multi-dimensional arrays
        // or, statically figuring out the parameter ordering

       // string entry_mapping = "";

        // what I want to do here is map from T to PROJECTED_DIMS.  I can do this by mapping each T entry to a number     

        // this won't work, logic is all wrong
        
  /*      int added = 0;
        //maybe a recursive function?
        static foreach(i, p ; PROJECTED_DIMS) {

            PROJECT_DIMS_OUTER:
            static foreach(j ; i .. T.length) {

                static if (typeid(p) == typeid(T[j])) {
                    
                    if (added > 0) {
                        entry_mapping ~= ",";
                    }
            
                    entry_mapping ~= "entry[" ~ i ~ "]";
                    added = added + 1;
                    break PROJECT_DIMS_OUTER;
                }    
            }
            
        }*/

        // wait, shit should I do this at runtime?  I can build tuples from others, so I just do this one at a time!
        // inside this foreach, a static foreach to generate the appropriate tuple = tuple!(tuple, entry[i])
        
        foreach (entry ; storage) {

            auto reduced_entry = entry;

            // generate lines of code to remove fields from reduced_entry

            auto REDUCED_DIMS = Reverse!(Reverse!(T));
            
            static foreach(i, p ; DIMS) {

                static if (staticIndexOf!( p, REDUCED_DIMS) == 0) {
                    reduced_entry = reduced_entry[1 .. $];

                } else {
                    reduced_entry = Tuple!(reduced_entry[0 .. staticIndexOf!( p, REDUCED_DIMS )] , reduced_entry[staticIndexOf!( p, REDUCED_DIMS ) + 1 .. $] );
                }

                REDUCED_DIMS = Erase!( p, REDUCED_DIMS );

            }
            
            
            newElements [ reduced_entry ] = true;
        }

        

        return new set!(NEWDIMS)(newElements.keys);
        
    }

    set!( removeLast!(DIMS) ) remove_dim_back(DIMS...)()
        if (DIMS.length > 0 && allSatisfy!(dimOfSet, DIMS)) 
    {

        alias NEWDIMS = removeLast!(DIMS);
        
        bool [Tuple!(NEWDIMS)] newElements;

        // go through storage, only adding unique tuples to newElements
        // this is harder than it needs to be, really would be easy with multi-dimensional arrays
        // or, statically figuring out the parameter ordering

       // string entry_mapping = "";

        // what I want to do here is map from T to PROJECTED_DIMS.  I can do this by mapping each T entry to a number     

        // this won't work, logic is all wrong
        
  /*      int added = 0;
        //maybe a recursive function?
        static foreach(i, p ; PROJECTED_DIMS) {

            PROJECT_DIMS_OUTER:
            static foreach(j ; i .. T.length) {

                static if (typeid(p) == typeid(T[j])) {
                    
                    if (added > 0) {
                        entry_mapping ~= ",";
                    }
            
                    entry_mapping ~= "entry[" ~ i ~ "]";
                    added = added + 1;
                    break PROJECT_DIMS_OUTER;
                }    
            }
            
        }*/



        template editTuple(FIRST, REMAIN ...) {
            static if (REMAIN.length == 0) {
                
            } else {

            }            
        }


        template MapTuple(I, DIMS, NEXT, TL...) {

            static if (TL.length == 0) {
                alias "" MapTuple;
                
            } else static if ( is(DIMS[0] == NEXT)) {
                
                alias "entry["~I~"], " ~ MapTuple!(I+1, DIMS[1..length], TL) MapTuple;
            } else {
                alias MapTuple!(I+1, DIMS, TL) MapTuple;
            }

        }
        
        // NO, you have one chance to define the tuple at runtime.
        
        foreach (entry ; storage) {

            auto reduced_entry = entry.reverse ;

            // generate lines of code to remove fields from reduced_entry

            alias REDUCED_DIMS = Reverse!(T);
            
            static foreach(i, p ; DIMS) {

                static if (staticIndexOf!( p, REDUCED_DIMS) == 0) {
                    reduced_entry = reduced_entry[1 .. $];

                } else {
                    reduced_entry = Tuple!(reduced_entry[0 .. staticIndexOf!( p, REDUCED_DIMS )] , reduced_entry[staticIndexOf!( p, REDUCED_DIMS ) + 1 .. $] );
                }

                REDUCED_DIMS = Erase!( p, REDUCED_DIMS );

            }
            
            
            newElements [ reduced_entry.reverse ] = true;
        }

        

        return new set!(NEWDIMS)(newElements.keys);
    }


    set!( AliasSeq!(T, A) ) cartesian_product(A) (set!(A) a) {

        alias NEWDIMS = AliasSeq!(T, A);

        Tuple!(NEWDIMS) [] newElements;  // TODO: Optimise this, we know how big this array should be

        foreach (Tuple!(T) mine; storage) {

            foreach (Tuple!(A) yours; a) {

                newElements ~= tuple(mine[], yours[]);
            }
        }

        return new set!(NEWDIMS)(newElements);
    }


    
    protected template dimOfSet(DIM) {
        enum dimOfSet = (staticIndexOf!(DIM, T) != -1);
    }

    protected template removeLast(FIRST, REMAIN ...) {
        static if (REMAIN.length > 0) {
            alias removeLast = Reverse!(Erase!(FIRST, Reverse!(  removeLast!( REMAIN ) )));
        } else {
            alias removeLast = Reverse!(Erase!(FIRST, Reverse!(T) ));
        }
    }

    protected template removeFirst(FIRST, REMAIN ...) {
        static if (REMAIN.length > 0) {    
            alias removeFirst = Erase!(FIRST, removeFirst!( REMAIN ) );
        } else {
            alias removeFirst = Erase!(FIRST, T);
        }
    }
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
}

@name("Set projection")
unittest {

    int size = 10;
    
    
    testObjSet testSet = new testObjSet(size);


    set!(testObj, testObj) newSet = testSet.cartesian_product(testSet);


    testObjSet finalSet = new testObjSet( newSet.orth_project!(testObj)() );

        
    assert(newSet.size() == size * size, "Set size is incorrect");

    foreach (obj ; newSet) {
        assert(newSet.contains(obj));
    }
}

