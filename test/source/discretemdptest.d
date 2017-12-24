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

class distribution(PARAMS...) : func!(double, PARAMS) {

    protected bool normalized;

    
    public this(set!PARAMS s, double def) {
        super(s, def);
        normalized = false;
    }

    public this(set!PARAMS s, double [Tuple!(PARAMS)] arr) {
        super(s, arr);
        normalized = false;
    }

    public this(set!PARAMS s, double [Tuple!(PARAMS)] arr, double def) {
        super(s, arr, def);
        normalized = false;
    }

    public this(set!PARAMS s, DistInitType init = DistInitType.None) {
        this(s, init, 10);
    }

    
    public this(set!PARAMS s, DistInitType init, double skewness) {
        normalized = false;
 
        if (init == DistInitType.None) {
            this(s, 0.0);
        } else {
            double [Tuple!(PARAMS)] arr;
 
            foreach(key ; s) {
                final switch(init) {
                  case DistInitType.Uniform:
                      arr[key] = 1.0;
                      break;
                  case DistInitType.Exponential:
                      import std.random;
                      arr[key] = exp(uniform01() * skewness);
                      break;
                  case DistInitType.RandomFromGaussian:
                      import std.random;
                      double total = 0;
                      for (int i = 0; i < 12; i ++)  // irwin-hall approximation of the normal distribution 
                           total += uniform01();
                      arr[key] = total;
                      break;
                  case DistInitType.None: // should never be here
                      break;

                }

            }

            arr.rehash();

            super(s, arr);

            normalize();
            
        }
    }


    public void normalize() {
        if (normalized) return;

        auto tot = 0.0;
        foreach(key ; mySet) {
            tot += storage.get(key, funct_default);
        }

        if (tot == 0.0) {
            throw new Exception("Empty distribution or all zero probabilities, cannot normalize");
        }

        foreach(key; mySet) {
            storage[key] = storage.get(key, funct_default) / tot;
        }

        normalized = true;
     }

     public bool isNormalized() {
        return normalized;
     }

     // will always return a sample
     public Tuple!(PARAMS) sample() {

        if (mySet.size() == 0) {
            throw new Exception("Cannot sample from zero sized distribution.");
        }
          
        normalize();

        import std.random;

        auto rand = uniform(0.0, 1.0);

        auto keys = mySet.toArray();
        randomShuffle(keys);

        auto mass = 0.0;
        foreach ( k; keys) {
            mass += storage.get(k, funct_default);

            if (mass >= rand)
                return k;
        }

        debug {
            import std.conv;
            throw new Exception("Didn't find a key to sample, ended at: " ~ to!string(mass) ~ " but wanted " ~ to!string(rand));
        } else {
            return keys[$-1];
        }

    }

    override protected void _postElementModified(Tuple!(PARAMS) key) {
        normalized = false;
    }

    double KLD(distribution!PARAMS other_dist) {
	
    	double returnval = 0;
    	foreach (i; mySet) {
            auto pr = storage.get(i, funct_default);
    		returnval += pr * log ( pr / other_dist[i]);
    	}
    	return returnval;
	
    }  

    double entropy() {
        double returnval = 0;

        foreach (i; mySet) {
            auto pr = storage.get(i, funct_default);            
            returnval += pr * log (pr);
        }
        return -returnval;

    }

    double crossEntropy(distribution!PARAMS other_dist) {
        return entropy() + KLD(other_dist);
    }

    void optimize() {
        foreach(key, val ; storage) {
            if (val == funct_default) {
                storage.remove(key);
            }
        }

        storage.rehash();
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


// Function and set experimenting
import std.traits;

class func(RETURN_TYPE, PARAM ...) {

    RETURN_TYPE [Tuple!(PARAM)] storage;
    set!PARAM mySet;

    RETURN_TYPE funct_default;

    public this(set!PARAM s, RETURN_TYPE def) {
        mySet = s;
        funct_default = def;
    }

    public this(set!PARAM s, RETURN_TYPE [Tuple!(PARAM)] arr) {
        mySet = s;
        storage = arr;
        foreach(key ; mySet) {
            funct_default = arr[key];
            break;
        }
    }

    public this(set!PARAM s, RETURN_TYPE [Tuple!(PARAM)] arr, RETURN_TYPE def) {
        mySet = s;
        storage = arr;
        funct_default = def;  
    }
    
    RETURN_TYPE opIndex(Tuple!(PARAM) i ) {
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

    RETURN_TYPE opIndex(PARAM i ) {
        return opIndex(tuple(i));
    }
    
    void opIndexAssign(RETURN_TYPE value, Tuple!(PARAM) i) {
          if ( mySet !is null && ! mySet.contains(i)) {
               throw new Exception("ERROR, key is not in the set this function is defined over.");
          }
          _preElementModified(i);
          storage[i] = value;
          _postElementModified(i);
    }

    void opIndexAssign(RETURN_TYPE value, PARAM i) {

        opIndexAssign(value, tuple(i));
    }

    // FOR NUMERIC RETURN TYPES ONLY
    void opIndexOpAssign(string op)(RETURN_TYPE rhs, Tuple!(PARAM) key) 
        if (isNumeric!(RETURN_TYPE))
    {
        RETURN_TYPE* p;
        p = (key in storage);
        if (p is null) {
            if ( mySet !is null && ! mySet.contains(key)) {
                throw new Exception("ERROR, key is not in the set this function is defined over.");
            }
            storage[key] = funct_default;
            p = (key in storage);
        }
        _preElementModified(key);
        mixin("*p " ~ op ~ "= rhs;");
        _postElementModified(key);
    }    

    void opIndexOpAssign(string op)(RETURN_TYPE rhs, PARAM key) 
        if (isNumeric!(RETURN_TYPE))
    {
        opIndexOpAssign!(op)(rhs, tuple(key));
        
    }
    // Since opIndexOpAssign must be non-virtual, we need callbacks that are virtual for subclasses to override behavior

    // called before an element's return value is modified
    protected void _preElementModified(Tuple!(PARAM) key) {
    }

    // called after an element's return value is modified
    protected void _postElementModified(Tuple!(PARAM) key) {
    }

    // operation with a same sized function (matrix op)
    func!(RETURN_TYPE, PARAM) opBinary(string op)(func!(RETURN_TYPE, PARAM) other) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "other[key];");
        }

        
        return new func!(RETURN_TYPE, PARAM)(mySet, result);
    }

    // operation with a single param function (vector op)
    func!(RETURN_TYPE, PARAM) opBinary(string op)(func!(RETURN_TYPE, PARAM[PARAM.length - 1]) other) 
        if (PARAM.length > 1 && (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/")))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            auto tempKey = tuple(key[key.length - 1]);
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "other[tempKey];");
        }

        
        return new func!(RETURN_TYPE, PARAM)(mySet, result);
    }

    // operation with a single value (scalar op)
    func!(RETURN_TYPE, PARAM) opBinary(string op)(RETURN_TYPE scalar) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "scalar;");
        }

        
        return new func!(RETURN_TYPE, PARAM)(mySet, result);
    }
    
    func!(RETURN_TYPE, PARAM) opBinaryRight(string op)(RETURN_TYPE scalar) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        return opBinary!(op)(scalar);
    }
        
    // These functions should probably stay removed, instead get the user to use the function's param set for looping:

    // foreach (key ; func.param_set)
    //      func[key] ... 
/*    auto byKey() {
        return storage.byKey();
    }

    auto byValue() {
        return storage.byValue();
    }

    auto byKeyValue() {
        return storage.byKeyValue();
    }

    public int opApply(scope int delegate(ref Tuple!(PARAM), ref RETURN_TYPE) dg) {
        int result = 0;
        foreach (key, value ; storage) {
            result = dg(key, value);
            if (result) break;

        }
        return result;
    }*/
    
    /*
    // I don't think this is needed, looping over just return values?
    public int opApply(scope int delegate(ref RETURN_TYPE) dg) {
        int result = 0;
        foreach (value ; storage) {
            result = dg(value);
            if (result) break;

        }
        return result;
    }
    */

    public set!(PARAM) param_set() {
        return mySet;
    }

    static if (PARAM.length == 1) {
        

        RETURN_TYPE max() {
       
            RETURN_TYPE max;
            bool setMax = false;
        
            foreach (key ; mySet) {

                RETURN_TYPE val = storage.get(key, funct_default);
            
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

                RETURN_TYPE val = storage.get(key, funct_default);
            
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

        RETURN_TYPE sumout()() 
            if (isNumeric!(RETURN_TYPE))
        {
       
            RETURN_TYPE sum = 0;
        
            foreach (key ; mySet) {

                sum += storage.get(key, funct_default);
            }

            return sum;

        }
        
    } else {
        
        func!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1] ) max()() {

            return max!(PARAM[PARAM.length - 1])();

        }


        func!(RETURN_TYPE, removeLast!(TOREMOVE) ) max(TOREMOVE...)() 
            if (TOREMOVE.length > 0 && allSatisfy!(dimOfSet, TOREMOVE))
        {
            alias SUBPARAM = removeLast!(TOREMOVE);

            auto newSet = mySet.orth_project!(SUBPARAM)();
        
            RETURN_TYPE [Tuple!(SUBPARAM)] max;


            template MapTuple(int I, int J, int K) {

                static if (I < 0) {
                    const char[] MapTuple = "";
                
                } else static if (J >= 0 && is(TOREMOVE[J] == PARAM[I])) {
                    const char[] MapTuple = MapTuple!(I-1, J-1, K);
                } else {
                    static if (K > 0) {
                        const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "combinedkey["~to!string(I)~"], ";
                    } else {
                        const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "combinedkey["~to!string(I)~"]";
                    }
                
                }

            }        
              

            foreach(combinedkey ; mySet) {
 
                RETURN_TYPE val = storage.get(combinedkey, funct_default);

                auto key = mixin( "tuple(" ~ MapTuple!(PARAM.length - 1, TOREMOVE.length - 1, 0) ~ ")" );

                RETURN_TYPE* p;
                p = (key in max);
                if (p is null) {
                    max[key] = val;
                } else {
                    if (val > *p) {
                        *p = val;
                    }
                }
                
            }
            

            return new func!(RETURN_TYPE,SUBPARAM)(newSet, max);
            
        }

        func!(Tuple!(PARAM[PARAM.length - 1]), PARAM[0 .. PARAM.length - 1] ) argmax()() {

            return argmax!(PARAM[PARAM.length - 1])();

        }
        
        func!(Tuple!(TOREMOVE), removeLast!(TOREMOVE) ) argmax(TOREMOVE...)() 
            if (TOREMOVE.length > 0 && allSatisfy!(dimOfSet, TOREMOVE))

        {
            alias SUBPARAM = removeLast!(TOREMOVE);

            auto newSet = mySet.orth_project!(SUBPARAM)();

            RETURN_TYPE [Tuple!(SUBPARAM)] max;
            Tuple!(TOREMOVE) [Tuple!(SUBPARAM)] max_key;
            
        
            template MapTuple(int I, int J, int K) {

                static if (I < 0) {
                    const char[] MapTuple = "";
                
                } else static if (J >= 0 && is(TOREMOVE[J] == PARAM[I])) {
                    const char[] MapTuple = MapTuple!(I-1, J-1, K);
                } else {
                    static if (K > 0) {
                        const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "combinedkey["~to!string(I)~"], ";
                    } else {
                        const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "combinedkey["~to!string(I)~"]";
                    }
                
                }

            } 

            template MapTuplePositive(int I, int J, int K) {

                static if (I < 0) {
                    const char[] MapTuplePositive = "";
                
                } else static if (J >= 0 && is(SUBPARAM[J] == PARAM[I])) {
                    static if (K > 0) {
                        const char[] MapTuplePositive =  MapTuplePositive!(I-1, J-1, K+1) ~ "combinedkey["~to!string(I)~"], ";
                    } else {
                        const char[] MapTuplePositive =  MapTuplePositive!(I-1, J-1, K+1) ~ "combinedkey["~to!string(I)~"]";
                    }
                } else {

                    const char[] MapTuplePositive = MapTuplePositive!(I-1, J, K);
                                
                }

            }             
            foreach(combinedkey ; mySet) {

                RETURN_TYPE val = storage.get(combinedkey, funct_default);
                
                auto key = mixin( "tuple(" ~ MapTuple!(PARAM.length - 1, TOREMOVE.length - 1, 0) ~ ")" );
                auto return_key = mixin( "tuple(" ~ MapTuplePositive!(PARAM.length - 1, SUBPARAM.length - 1, 0) ~ ")" );

                RETURN_TYPE* p;
                p = (key in max);
                if (p is null) {
                    max[key] = val;
                    max_key[key] = return_key;
                } else {
                    if (val > *p) {
                        *p = val;
                        max_key[key] = return_key;
                    }
                }

                
            }

            return new func!(Tuple!(TOREMOVE),SUBPARAM)(newSet, max_key);
        }


        
        func!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1] ) sumout()() 
            if (isNumeric!(RETURN_TYPE))
        {

            return sumout!(PARAM[PARAM.length - 1])();

        }


        func!(RETURN_TYPE, removeLast!(TOREMOVE) ) sumout(TOREMOVE...)() 
            if (TOREMOVE.length > 0 && allSatisfy!(dimOfSet, TOREMOVE) && isNumeric!(RETURN_TYPE))
        {
            alias SUBPARAM = removeLast!(TOREMOVE);

            auto newSet = mySet.orth_project!(SUBPARAM)();
        
            RETURN_TYPE [Tuple!(SUBPARAM)] sum;


            template MapTuple(int I, int J, int K) {

                static if (I < 0) {
                    const char[] MapTuple = "";
                
                } else static if (J >= 0 && is(TOREMOVE[J] == PARAM[I])) {
                    const char[] MapTuple = MapTuple!(I-1, J-1, K);
                } else {
                    static if (K > 0) {
                        const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "combinedkey["~to!string(I)~"], ";
                    } else {
                        const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "combinedkey["~to!string(I)~"]";
                    }
                
                }

            }        
              

            foreach(combinedkey ; mySet) {

                RETURN_TYPE val = storage.get(combinedkey, funct_default);
                
                auto key = mixin( "tuple(" ~ MapTuple!(PARAM.length - 1, TOREMOVE.length - 1, 0) ~ ")" );

                RETURN_TYPE* p;
                p = (key in sum);
                if (p is null) {
                    sum[key] = val;
                } else {
                    *p += val;
                }
                
            }
            

            return new func!(RETURN_TYPE,SUBPARAM)(newSet, sum);
            
        }       


        
        func!(RETURN_TYPE, PARAM[0..PARAM.length - 1]) apply()(func!(Tuple!(PARAM[PARAM.length - 1]), PARAM[0..PARAM.length -1]) f)
        {

            RETURN_TYPE [ Tuple!(PARAM[0..PARAM.length -1]) ] chosen;

            foreach( b_key ; f.param_set()) {

                auto newKey = tuple(b_key[], f[b_key][0] );

                chosen[b_key] = storage[newKey];

            }

            return new func!(RETURN_TYPE, PARAM[0..PARAM.length -1])(f.param_set(), chosen);        

        }
    }


    override string toString() {

        string returnval = "";

        foreach (key ; mySet) {
            auto val = storage.get(key, funct_default);

            returnval ~= to!string(key) ~ " => " ~ to!string(val) ~ ", ";
        }
        
        return returnval;
    }

    
    protected template removeLast(FIRST, T ...) {
        static if (T.length > 0) {
            alias removeLast = Reverse!(Erase!(FIRST, Reverse!(  removeLast(T) )));
        } else {
            alias removeLast = Reverse!(Erase!(FIRST, Reverse!(PARAM)));
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

    public Tuple!(T) [] toArray() {
        return storage.dup;
    }
    public int opApply(int delegate(ref Tuple!(T)) dg) {
          int result = 0;
          foreach (value ; storage) {
               result = dg(value);
               if (result) break;

          }
          return result;
    }
    
    set!(PROJECTED_DIMS) orth_project(PROJECTED_DIMS...)()
        if (PROJECTED_DIMS.length > 0 && allSatisfy!(dimOfSet, PROJECTED_DIMS) ) 
    {
        static if (is (PROJECTED_DIMS == T)) {
            return new set!(T)(storage.dup);
        } else {
            return remove_dim_back!( removeFirst!(PROJECTED_DIMS) )();
        }
    }

    set!( removeFirst!(DIMS) ) remove_dim_front(DIMS...)()
        if (DIMS.length > 0 && allSatisfy!(dimOfSet, DIMS) 
            && removeFirst!(DIMS).length == (T.length - DIMS.length) && DIMS.length < T.length
            && dimOrderingCorrectForward!(DIMS.length, DIMS, T)) 
    {

        alias NEWDIMS = removeFirst!(DIMS);
        
        bool [Tuple!(NEWDIMS)] newElements;


        template MapTuple(int I, int J, int K) {

            static if (I >= T.length) {
                const char[] MapTuple = "";
                
            } else static if (J < DIMS.length && is(DIMS[J] == T[I])) {
                const char[] MapTuple = MapTuple!(I+1, J+1, K);
            } else {
                static if (K > 0) {
                    const char[] MapTuple =  ", entry["~to!string(I)~"]" ~ MapTuple!(I+1, J, K+1);
                } else {
                    const char[] MapTuple =  "entry["~to!string(I)~"]" ~ MapTuple!(I+1, J, K+1);
                }
                
            }

        }        
        
        foreach (entry ; storage) {
            
            newElements [ mixin("tuple(" ~ MapTuple!(0, 0, 0) ~ ")" ) ] = true;
        }

        

        return new set!(NEWDIMS)(newElements.keys);
        
    }

    set!( removeLast!(DIMS) ) remove_dim_back(DIMS...)()
        if (DIMS.length > 0 && allSatisfy!(dimOfSet, DIMS) && removeLast!(DIMS).length == (T.length - DIMS.length)
        && DIMS.length < T.length && dimOrderingCorrectBackward!(DIMS.length, DIMS, T)) 
    {

        alias NEWDIMS = removeLast!(DIMS);
        
        bool [Tuple!(NEWDIMS)] newElements;

        template MapTuple(int I, int J, int K) {

            static if (I < 0) {
                const char[] MapTuple = "";
                
            } else static if (J >= 0 && is(DIMS[J] == T[I])) {
                const char[] MapTuple = MapTuple!(I-1, J-1, K);
            } else {
                static if (K > 0) {
                    const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "entry["~to!string(I)~"], ";
                } else {
                    const char[] MapTuple =  MapTuple!(I-1, J, K+1) ~ "entry["~to!string(I)~"]";
                }
                
            }

        }

        foreach (entry ; storage) {

            newElements [ mixin("tuple(" ~ MapTuple!(T.length - 1, DIMS.length - 1, 0) ~ ")" ) ] = true;
        }

        

        return new set!(NEWDIMS)(newElements.keys);
    }


    set!( AliasSeq!(T, A) ) cartesian_product(A) (set!(A) a) {

        alias NEWDIMS = AliasSeq!(T, A);

        Tuple!(NEWDIMS) [] newElements;

        newElements.length = storage.length * a.storage.length;

        size_t i = 0;
        
        foreach (Tuple!(T) mine; storage) {

            foreach (Tuple!(A) yours; a) {

                newElements[i] = tuple(mine[], yours[]);
                i ++;
            }
        }

        return new set!(NEWDIMS)(newElements);
    }


    
    protected template dimOfSet(DIM) {
        enum dimOfSet = (staticIndexOf!(DIM, T) != -1);
    }


    // ALLP[0] is the split between dimensions to remove and the remainder of T
    protected template dimOrderingCorrectForward(ALLP...) {
        enum I = ALLP[0];
        alias FIRST = ALLP[1];
        alias REVISEDT = ALLP[I+1..ALLP.length];

        static if (I > 1) {
            alias NEXT = ALLP[2];
            alias REMAINDER = ALLP[2..I+1];
            enum dimOrderingCorrectForward = staticIndexOf!(FIRST, REVISEDT) <= staticIndexOf!(NEXT, Erase!(FIRST, REVISEDT)) &&
                    dimOrderingCorrectForward!(I-1, REMAINDER, Erase!(FIRST, REVISEDT));

        } else {
            enum dimOrderingCorrectForward = staticIndexOf!(FIRST, REVISEDT) != -1;

        }
    }    

    protected template dimOrderingCorrectBackward(ALLP...) {

        enum I = ALLP[0];
        alias REVISEDT = ALLP[I+1..ALLP.length];
        
        enum dimOrderingCorrectBackward = dimOrderingCorrectForward!(I, Reverse!(ALLP[1..I+1]), Reverse!(REVISEDT));
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
    
