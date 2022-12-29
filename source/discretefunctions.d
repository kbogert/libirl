module discretefunctions;

// Discrete functions and sets
import std.traits;
import std.typetuple;
import std.typecons;
import std.conv;
import std.math;
import std.random;


// could be optimized for 1D sets by removing the tuples, if I feel like it I guess
class Set(T ...) {

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

    public bool contains(T i) {
        return contains(tuple(i));
    }

    public Tuple!(T) [] toArray() {
        return storage.dup;
    }

    public Tuple!(T) getOne() {
        if (storage.length == 0)
            throw new Exception("Cannot get an element from an empty set.");
        return storage[0];
    }
    
    public int opApply(int delegate(ref Tuple!(T)) dg) {
          int result = 0;
          foreach (value ; storage) {
               result = dg(value);
               if (result) break;

          }
          return result;
    }
    
    Set!(PROJECTED_DIMS) orth_project(PROJECTED_DIMS...)()
        if (PROJECTED_DIMS.length > 0 && allSatisfy!(dimOfSet, PROJECTED_DIMS) ) 
    {
        static if (is (PROJECTED_DIMS == T)) {
            return new Set!(T)(storage.dup);
        } else {
            return remove_dim_back!( removeFirst!(PROJECTED_DIMS) )();
        }
    }

    Set!( removeFirst!(DIMS) ) remove_dim_front(DIMS...)()
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

        

        return new Set!(NEWDIMS)(newElements.keys);
        
    }

    Set!( removeLast!(DIMS) ) remove_dim_back(DIMS...)()
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

        

        return new Set!(NEWDIMS)(newElements.keys);
    }


    Set!( AliasSeq!(T, A) ) cartesian_product(A ...) (Set!(A) a) {

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

        return new Set!(NEWDIMS)(newElements);
    }
        
    public Set!(T) unionWith (Set!(T) other_set) {

        Tuple!(T) [] newElements = storage.dup;

        foreach(newElement ; other_set.storage) {
            bool is_in_already = false;

            foreach( check ; newElements) {
                if (check == newElement) {
                    is_in_already = true;
                    break;
                }
            }

            if (! is_in_already) 
                newElements ~= newElement;
                
        }
        
        return new Set!(T)(newElements);
        
    }

    public Set!(T) intersectionWith (Set!(T) other_set) {
        Tuple!(T) [] newElements;

        foreach(newElement ; other_set.storage) {
            bool is_in_both = false;

            foreach( check ; storage) {
                if (check == newElement) {
                    is_in_both = true;
                    break;
                }
            }

            if (is_in_both) 
                newElements ~= newElement;
                
        }
        return new Set!(T)(newElements);
    }

    public Set!(T) differenceWith (Set!(T) other_set) {
        
        Set!(T) subSet = intersectionWith(other_set);
        Tuple!(T) [] newElements;
        
        foreach(newElement ; storage) {
            bool is_in_both = false;

            foreach( check ; subSet.storage) {
                if (check == newElement) {
                    is_in_both = true;
                    break;
                }
            }

            if (!is_in_both) 
                newElements ~= newElement;
                
        }
        return new Set!(T)(newElements);
    }

    
    public Set!( Reverse!(T) ) reverse_params () {

        Tuple!(Reverse!(T)) [] newElements = new Tuple!(Reverse!(T)) [storage.length];

        size_t i = 0;
        
        foreach ( entry; storage ) {
            newElements[i] = entry.reverse;
            i ++;
        }
                
        return new Set!(Reverse!(T))(newElements);
        
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


class Function (RETURN_TYPE, PARAM ...) {

    RETURN_TYPE [Tuple!(PARAM)] storage;
    Set!PARAM mySet;

    RETURN_TYPE funct_default;

    public this(Set!PARAM s, RETURN_TYPE def) {
        mySet = s;
        funct_default = def;
    }

    public this(Set!PARAM s, RETURN_TYPE [Tuple!(PARAM)] arr) {
        mySet = s;
        storage = arr;
        foreach(key ; mySet) {
            funct_default = arr[key];
            break;
        }
    }

    public this(Set!PARAM s, RETURN_TYPE [Tuple!(PARAM)] arr, RETURN_TYPE def) {
        mySet = s;
        storage = arr;
        funct_default = def;  
    }

    public this(Function!(RETURN_TYPE, PARAM) toCopy) {
        storage = toCopy.storage.dup;
        mySet = toCopy.mySet;
        funct_default = toCopy.funct_default;
    }
    
    RETURN_TYPE opIndex(Tuple!(PARAM) i ) {
        RETURN_TYPE* p;
        p = (i in storage);
        if (p !is null) {
            return *p;
        }
        if ( mySet !is null && ! mySet.contains(i)) {
            throw new Exception("ERROR, key " ~ to!string(i) ~ "is not in the set this function is defined over.");
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
    Function!(RETURN_TYPE, PARAM) opBinary(string op)(Function!(RETURN_TYPE, PARAM) other) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "other[key];");
        }

        
        return new Function!(RETURN_TYPE, PARAM)(mySet, result);
    }

    // operation with a single param function (vector op)
    Function!(RETURN_TYPE, PARAM) opBinary(string op)(Function!(RETURN_TYPE, PARAM[PARAM.length - 1]) other) 
        if (PARAM.length > 1 && (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/")))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            auto tempKey = tuple(key[key.length - 1]);
            mixin("auto tempResult = storage.get(key, funct_default) " ~ op ~ "other[tempKey];");
            if (tempResult != funct_default) {
                result[key] = tempResult;
            }
        }

        
        return new Function!(RETURN_TYPE, PARAM)(mySet, result, funct_default);
    }

    // operation with a single value (scalar op)
    Function!(RETURN_TYPE, PARAM) opBinary(string op)(RETURN_TYPE scalar) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "scalar;");
        }

        
        return new Function!(RETURN_TYPE, PARAM)(mySet, result);
    }
    
    Function!(RETURN_TYPE, PARAM) opBinaryRight(string op)(RETURN_TYPE scalar) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        return opBinary!(op)(scalar);
    }
        
    // These functions should probably stay removed, instead get the user to use the function's param set for looping:

    // foreach (key ; Function.param_set)
    //      Function[key] ... 
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

    
    static if (isNumeric!(RETURN_TYPE)) {
        Function!(RETURN_TYPE, PARAM) abs() 
        {

            RETURN_TYPE [Tuple!(PARAM)] result;

            foreach (key ; mySet) {
                result[key] = std.math.abs(storage.get(key, funct_default));
            }

        
            return new Function!(RETURN_TYPE, PARAM)(mySet, result);        

        }
    }
    
    size_t opDollar(size_t pos)() {
        return mySet.size();
    }

    size_t size() {
        return mySet.size();
    }
     
    public Set!(PARAM) param_set() {
        return mySet;
    }

    public Function!(RETURN_TYPE, Reverse!(PARAM)) reverse_params() {

        RETURN_TYPE [Tuple!(Reverse!(PARAM))] vals;

        foreach (entry, val ; storage) {

            vals[ entry.reverse ] = val;
        }
            
        auto newset = mySet.reverse_params();
        
        return new Function!(RETURN_TYPE, Reverse!(PARAM))(newset, vals, funct_default);
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

        RETURN_TYPE min() {
       
            RETURN_TYPE min;
            bool setMin = false;
        
            foreach (key ; mySet) {

                RETURN_TYPE val = storage.get(key, funct_default);
            
                if (! setMin ) {
                    min = val;
                    setMin = true;
                } else {
                    if (val < min) {
                        min = val;
                    }
                }                
            
            }

            return min;

        }
        
        // numerically stable softmax
        RETURN_TYPE softmax()() 
            if (isFloatingPoint!(RETURN_TYPE))
        {
            RETURN_TYPE smax = 0;
            auto max = max();

            foreach (key ; mySet) {

                RETURN_TYPE val = storage.get(key, funct_default);

                smax += exp(val - max);
            }            

            if (smax == 0)
                smax = RETURN_TYPE.min_normal;
                
            return max + log(smax);
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
        Tuple!(PARAM) argmax_shuffled() {
            RETURN_TYPE max;
            Tuple!(PARAM) max_param;
            bool setMax = false;
            import std.random;

            foreach (key ; mySet.toArray.randomShuffle()) {

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
        
        Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1] ) max()() {

            return max!(PARAM[PARAM.length - 1])();

        }


        Function!(RETURN_TYPE, removeLast!(TOREMOVE) ) max(TOREMOVE...)() 
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
            

            return new Function!(RETURN_TYPE,SUBPARAM)(newSet, max);
            
        }

        
        // numerically stable softmax
        Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1] ) softmax()() {

            return softmax!(PARAM[PARAM.length - 1])();

        }

        Function!(RETURN_TYPE, removeLast!(TOREMOVE) ) softmax(TOREMOVE...)() 
            if (TOREMOVE.length > 0 && allSatisfy!(dimOfSet, TOREMOVE) && isFloatingPoint!(RETURN_TYPE))
        {
            alias SUBPARAM = removeLast!(TOREMOVE);

            auto newSet = mySet.orth_project!(SUBPARAM)();
        
            RETURN_TYPE [Tuple!(SUBPARAM)] smax;


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

            auto mmax = max!(TOREMOVE)();

            foreach(combinedkey ; mySet) {
 
                RETURN_TYPE val = storage.get(combinedkey, funct_default);

                auto key = mixin( "tuple(" ~ MapTuple!(PARAM.length - 1, TOREMOVE.length - 1, 0) ~ ")" );

                RETURN_TYPE* p;
                p = (key in smax);
                if (p is null) {
                    smax[key] = exp(val - mmax[key]);
                } else {
                    *p += exp(val - mmax[key]);
                }
                
            }

            foreach ( key, ref val; smax) {
                if (val > 0)
                    val = mmax[key] + log(val);
            }            

            return new Function!(RETURN_TYPE,SUBPARAM)(newSet, smax);
        }        

        Function!(Tuple!(PARAM[PARAM.length - 1]), PARAM[0 .. PARAM.length - 1] ) argmax()() {

            return argmax!(PARAM[PARAM.length - 1])();

        }
        
        Function!(Tuple!(TOREMOVE), removeLast!(TOREMOVE) ) argmax(TOREMOVE...)() 
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
                
                } else static if (J >= 0 && is(TOREMOVE[J] == PARAM[I])) {
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
                
                Tuple!(SUBPARAM) key = mixin( "tuple(" ~ MapTuple!(PARAM.length - 1, TOREMOVE.length - 1, 0) ~ ")" );
                Tuple!(TOREMOVE) return_key = mixin( "tuple(" ~ MapTuplePositive!(PARAM.length - 1, TOREMOVE.length - 1, 0) ~ ")" );

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

            return new Function!(Tuple!(TOREMOVE),SUBPARAM)(newSet, max_key);
        }


        
        Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1] ) sumout()() 
            if (isNumeric!(RETURN_TYPE))
        {

            return sumout!(PARAM[PARAM.length - 1])();

        }


        Function!(RETURN_TYPE, removeLast!(TOREMOVE) ) sumout(TOREMOVE...)() 
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

            return new Function!(RETURN_TYPE,SUBPARAM)(newSet, sum);
            
        }       


        
        Function!(RETURN_TYPE, PARAM[0..PARAM.length - 1]) apply()(Function!(Tuple!(PARAM[PARAM.length - 1]), PARAM[0..PARAM.length -1]) f)
        {

            RETURN_TYPE [ Tuple!(PARAM[0..PARAM.length -1]) ] chosen;

            foreach( b_key ; f.param_set()) {

                auto newKey = tuple(b_key[], f[b_key][0] );

                chosen[b_key] = storage.get(newKey, funct_default);

            }

            return new Function!(RETURN_TYPE, PARAM[0..PARAM.length -1])(f.param_set(), chosen);        

        }
    }


    override string toString() {

        string returnval = "";

        foreach (key ; mySet) {
            auto val = storage.get(key, funct_default);

            foreach (k ; key) {
                static if (isTuple!(typeof(k))) {
                    returnval ~= "(";
                    foreach(k2 ; k) {
                        returnval ~= to!string(k2) ~ ", ";
                    }
                    returnval ~= ") ";
                } else {
                    returnval ~= to!string(k) ~ ", ";
                }
            }
            static if (isTuple!(typeof(val))) {
                returnval ~= " => ";
                foreach (v ; val)
                    returnval ~= to!string(v) ~ ", ";

            } else {
                returnval ~= " => " ~ to!string(val) ~ ", ";
            }
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

// convenience functions
public auto max(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.max();
}

public auto min(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.min();
}

public auto max(OVER, RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.max!(OVER)();
}

public auto softmax(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.softmax();
}

public auto softmax(OVER, RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.softmax!(OVER)();
}

public auto sumout(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.sumout();
}

public auto sumout(OVER, RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.sumout!(OVER)();
}

public auto argmax(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.argmax();
}

public auto argmax(OVER, RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.argmax!(OVER)();
}

public auto argmax_shuffled(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.argmax_shuffled();
}

public auto argmax_shuffled(OVER, RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.argmax_shuffled!(OVER)();
}

public auto abs(RETURN_TYPE, PARAM...) (Function!(RETURN_TYPE, PARAM) f) {

    return f.abs();
}


enum DistInitType {None, Uniform, Exponential, RandomFromGaussian};


class Distribution(PARAMS...) : Function!(double, PARAMS) {

    protected bool normalized;
    protected double normal;

    public this(Function!(double, PARAMS) init) {
        super(init);
        normalized = false;
    }

    
    public this(Set!PARAMS s, double def) {
        super(s, def);
        normalized = false;
    }

    public this(Set!PARAMS s, double [Tuple!(PARAMS)] arr) {
        super(s, arr);
        normalized = false;
    }

    public this(Set!PARAMS s, double [Tuple!(PARAMS)] arr, double def) {
        super(s, arr, def);
        normalized = false;
    }

    public this(Set!PARAMS s, DistInitType init = DistInitType.None) {
        this(s, init, 10);
    }

    
    public this(Set!PARAMS s, DistInitType init, double skewness) {
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

            // arr = arr.rehash();

            super(s, arr);

            normalize();
            
        }
    }
/*
    // division with a distribution over some of the parameters, Pr(A , B) / Pr(B) = Pr (A | B) and Pr(A , B) / Pr(A) = Pr (B | A)
    ConditionalDistribution!( removeFirst!(DIMS), DIMS ) opBinary(string op, DIMS...)(Distribution!(DIMS) other)
        if (DIMS.length > 0 && (op="/") && allSatisfy!(dimOfSet, DIMS) && removeFirst!(DIMS).length == (PARAMS.length - DIMS.length)
        && DIMS.length < PARAMS.length && dimOrderingCorrectForward!(DIMS.length, DIMS, PARAMS)) 
    {

    }
*/
    
    public double normalize() {
        if (normalized) return normal;

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
        normal = tot;
        return tot;
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
            assert(0);
        } else {
            return keys[$-1];
        }

    }

    override protected void _postElementModified(Tuple!(PARAMS) key) {
        normalized = false;
    }

    double KLD(Distribution!PARAMS other_dist) {
	
    	double returnval = 0;
    	foreach (i; mySet) {
            auto pr = storage.get(i, funct_default);
            if (pr != 0.0 && pr != -0.0)            
    		  returnval += pr * log ( pr / other_dist[i]);
    	}
    	return returnval;
	
    }  

    double entropy() {
        double returnval = 0;

        foreach (i; mySet) {
            auto pr = storage.get(i, funct_default);
            if (pr != 0.0 && pr != -0.0)            
                returnval += pr * log (pr);
        }
        return -returnval;

    }

    double crossEntropy(Distribution!PARAMS other_dist) {
        return entropy() + KLD(other_dist);
    }

    double JSD(Distribution!PARAMS other_dist) {

        // find the middle "distribution"
        Distribution!PARAMS M = new Distribution!PARAMS(mySet, 0.0);
        foreach(i; mySet) {
            M[i] = 0.5*(storage.get(i, funct_default) + other_dist[i]);
        }

        return 0.5*KLD(M) + 0.5*(other_dist.KLD(M));

    }

    void optimize() {
        foreach(key, val ; storage) {
            if (val == funct_default) {
                storage.remove(key);
            }
        }

        //storage = storage.rehash();
    }
}

// a specialized distribution, basically a function that returns distributions
// only need a conditional distributions over single sets right now

template ConditionalDistribution(OVER, PARAMS...) {

    // This flattens out the conditional distribution, but I'm not sure I want to do so
//static if (isTuple!(OVER)) {
//    alias OVER_2 = OVER.Types;
//} else {
    alias OVER_2 = OVER;
//}
    
class ConditionalDistribution : Function!(Distribution!(OVER_2), PARAMS) 
{

    // need this to create distributions, for instance when accessing a param set that hasn't already been inserted
    Set!(OVER_2) myOverSet;
    
    public this (Distribution!OVER_2 [Tuple!(PARAMS)] init, Set!(PARAMS) param_set) {

        if (init.length == 0) {
            throw new Exception("Initial distribution cannot be empty");
        }
        auto key = init.keys()[0];
        myOverSet = init[key].param_set();
        
        super(param_set, init);
    }

    public this (Set!(OVER_2) over_params, Set!(PARAMS) param_set) 
    {
        myOverSet = over_params;
        super(param_set, new Distribution!(OVER_2)(over_params));
    }

    public this (Set!(OVER_2) over_params, Set!(PARAMS) param_set, double [Tuple!(OVER_2)] [Tuple!(PARAMS)] init) {

        Distribution!(OVER_2) [Tuple!(PARAMS)] builtDistr;

        foreach(key, val ; init) {
            builtDistr[key] = new Distribution!(OVER_2)(over_params, val);
        }
        this(builtDistr, param_set);
        
    }

    public Set!(OVER_2) over_param_set() {
        return myOverSet;
    }


    // converts this structured function into a flat one that doesn't have any distribution features
    public Function!(double, PARAMS, OVER_2) flatten() {

        auto combined_params = param_set.cartesian_product(myOverSet);


        double [Tuple!(PARAMS, OVER_2)] tempArray;
        
        foreach(key1; param_set) {
            Distribution!(OVER_2)* p;
            p = (key1 in storage);
            if (p ! is null) {
                foreach(key2; myOverSet) {
                    auto fullKey = tuple(key1[], key2[]);

//                    tempArray[fullKey] = 0.0;
//                } else {
                    auto element = (*p)[key2];
                    if (element != 0.0)
                       tempArray[fullKey] = element;
                }
            }
        }

        return new Function!(double, PARAMS, OVER_2)(combined_params, tempArray, 0.0);
    }

    // operation with a same sized function (matrix op)
    Function!(double, PARAMS, OVER_2) opBinary(string op)(Function!(double, PARAMS, OVER_2) other) 
        if (PARAMS.length > 0 && (op=="+"||op=="-"||op=="*"||op=="/"))
    {
        return flatten().opBinary!(op)(other);
    }

    // multiplication with a distribution over the parameters, Pr(A | B) * Pr(B) = Pr(A , B)
    // The next function proved more popular (Pr(A | B) * Pr(B) = Pr(B , A) )
/*    Distribution!(OVER, PARAMS) opBinary(string op)(Distribution!(PARAMS) other) 
        if (PARAMS.length > 0 && (op=="*"))
    {
        Distribution!(OVER, PARAMS) returnval = new Distribution!(OVER, PARAMS)(over_param_set.cartesian_product(mySet), 0.0);

        foreach (key1 ; mySet) {

            Distribution!(OVER)* p;
            p = (key1 in storage);
            if (p ! is null) {
                foreach (key2; over_param_set) {

                    auto fullkey = tuple(key2[], key1[]);

                    auto element = (*p)[key2];

                    returnval[fullkey] = element * other[key1];
                
                }
            }
        }

        return returnval;
    }
*/

    // multiplication with a distribution over the parameters, Pr(A | B) * Pr(B) = Pr(B , A)
    Distribution!(PARAMS, OVER_2) opBinary(string op)(Distribution!(PARAMS) other) 
        if (PARAMS.length > 0 && (op=="*"))
    {
        double [Tuple!(PARAMS, OVER_2)] arr;
        
        foreach (key1 ; mySet) {

            Distribution!(OVER_2)* p;
            p = (key1 in storage);
            if (p ! is null) {
                foreach (key2; myOverSet) {

                    auto fullkey = tuple(key1[], key2[]);

                    auto element = (*p)[key2];
                    auto o = other[key1];
                    
                    if (element != 0.0 && o != 0.0)
                        arr[fullkey] = element * o;
                
                }
            }
        }

        return new Distribution!(PARAMS, OVER_2)(mySet.cartesian_product(myOverSet), arr, 0.0);

    }
    
    // operation with the over params function (vector op)
    Function!(double, PARAMS, OVER_2) opBinary(string op)(Function!(double, OVER_2) other) 
        if (PARAMS.length > 0 && ((op=="+"||op=="-"||op=="*"||op=="/")))
    {
        return flatten().opBinary!(op)(other);
    }

    // operation with a single value (scalar op)
    Function!(RETURN_TYPE, PARAMS, OVER_2) opBinary(string op)(double scalar) 
        if ((op=="+"||op=="-"||op=="*"||op=="/"))
    {
        auto combined_params = param_set.cartesian_product(myOverSet);

        Function!(double, PARAMS, OVER_2) returnval = new Function!(double, PARAMS, OVER_2)(combined_params);

        foreach(key; param_set) {
            foreach(key2; myOverSet) {
                auto fullKey = tuple(key1[], key2[]);

                mixin("returnval[fullkey] = storage.get(key1, new Distribution!(OVER_2)(over_params))[key2] " ~ op ~ " scalar;");
            }
        }

        return returnval;        
    }
    
    Function!(RETURN_TYPE, PARAMS, OVER_2) opBinaryRight(string op)(double scalar) 
        if ((op=="+"||op=="-"||op=="*"||op=="/"))
    {

        return opBinary!(op)(scalar);
    }

    override Distribution!(OVER_2) opIndex(Tuple!(PARAMS) i ) {
        Distribution!(OVER_2) * p;
        p = (i in storage);
        if (p !is null) {
            return *p;
        }
        if ( mySet !is null && ! mySet.contains(i)) {
            throw new Exception("ERROR, key is not in the set this function is defined over.");
        }
        storage[i] = new Distribution!(OVER_2)(myOverSet);
        return storage[i];
    }

}
}
/*
class ConditionalDistribution(SPLIT, PARAMS...) : Function!(Distribution!(PARAMS[0..SPLIT]), PARAMS[SPLIT..PARAMS.length]) {

    alias OVER = PARAMS[0..SPLIT];



    
    // converts this structured function into a flat one that doesn't have any distribution features
    public Function!(double, PARAMS[SPLIT..PARAMS.length], PARAMS[0..SPLIT]) flatten() {

        

    }

    // operation with a same sized function (matrix op)
    Function!(double, PARAMS[SPLIT..PARAMS.length], PARAMS[0..SPLIT]) opBinary(string op)(Function!(double, PARAMS[SPLIT..PARAMS.length], PARAMS[0..SPLIT]) other) 
        if ((op=="+"||op=="-"||op=="*"||op=="/"))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "other[key];");
        }

        
        return new Function!(RETURN_TYPE, PARAM)(mySet, result);
    }

    // operation with the over params function (vector op)
    Function!(double, PARAMS[SPLIT..PARAMS.length], PARAMS[0..SPLIT]) opBinary(string op)(Function!(double, OVER) other) 
        if (PARAM.length > 1 && (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/")))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            auto tempKey = tuple(key[key.length - 1]);
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "other[tempKey];");
        }

        
        return new Function!(RETURN_TYPE, PARAM)(mySet, result);
    }

    // operation with a single value (scalar op)
    Function!(RETURN_TYPE, PARAMS[SPLIT..PARAMS.length], PARAMS[0..SPLIT]) opBinary(string op)(double scalar) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        RETURN_TYPE [Tuple!(PARAM)] result;

        foreach (key ; mySet) {
            mixin("result[key] = storage.get(key, funct_default) " ~ op ~ "scalar;");
        }

        
        return new Function!(RETURN_TYPE, PARAM)(mySet, result);
    }
    
    Function!(RETURN_TYPE, PARAMS[SPLIT..PARAMS.length], PARAMS[0..SPLIT]) opBinaryRight(string op)(RETURN_TYPE scalar) 
        if (isNumeric!(RETURN_TYPE) && (op=="+"||op=="-"||op=="*"||op=="/"))
    {

        return opBinary!(op)(scalar);
    }
    
    
}
*/

// a specialized set of params X timesteps
class Sequence (PARAMS...) {

    Tuple!(PARAMS) [] timesteps;

    public this(Tuple!(PARAMS) [] t) {
        timesteps = t;
    }

    public this() {
    }

    public this(size_t size) {
        timesteps.length = size;
    }

    
    public Tuple!(PARAMS) opIndex(size_t idx) {

        return timesteps[idx];        
    }

    public Tuple!(PARAMS) [] opIndex() {
        return timesteps[];
    }

    // slicing support
    public Sequence!(PARAMS) opIndex(size_t[2] start) {

        return new Sequence( timesteps[start[0] .. start[1]].dup );
    }

    public size_t[2] opSlice(size_t dim)(size_t start, size_t end) 
        if (dim == 0)
    in { assert(start >= 0 && end <= this.opDollar!dim); }
    body
    {
        return [start, end];
    }

    void opIndexAssign(Tuple!(PARAMS) value, size_t i) {
          timesteps[i] = value;
    }


    public void opOpAssign(string op)(Tuple!(PARAMS) addition) 
        if (op == "~")
    {

        timesteps ~= addition;
    }

    public void opOpAssign(string op)(Sequence!PARAMS addition) 
        if (op == "~")
    {

        timesteps ~= addition.timesteps;
    }

    public Sequence!(PARAMS) opBinary(string op)(Tuple!(PARAMS) other) 
        if (op == "~" )
    {

        auto temp = timesteps.dup;
        temp ~= other;
        
        return new Sequence!(PARAMS)( temp );
    }

    public Sequence!(PARAMS) opBinary(string op)(Sequence!(PARAMS) other) 
        if (op == "~" )
    {

        auto temp = timesteps.dup;
        temp ~= other.timesteps.dup;
        
        return new Sequence!(PARAMS)( temp );
    }
        
    public int opApply(scope int delegate(ref size_t, ref Tuple!(PARAMS)) dg) {
        int result = 0;
        foreach (key, value ; timesteps) {
            result = dg(key, value);
            if (result) break;

        }
        return result;
    }
            
    public int opApply(scope int delegate(ref Tuple!(PARAMS)) dg) {
        int result = 0;
        foreach (value ; timesteps) {
            result = dg(value);
            if (result) break;

        }
        return result;
    }
    

    
    size_t opDollar(size_t pos)() {
        return timesteps.length - 1;
    }

    size_t length() {
        return timesteps.length;
    }        

    public void setLength(size_t l) {
        timesteps.length = l;
    }
    
    override string toString() {

        string returnval = "";

        foreach (timestep ; timesteps) {

            
            returnval ~= "( ";
            foreach (i, portion ; timestep ) {
                returnval ~= to!string(portion);
                if (i < timestep.length - 1) {
                    returnval ~= " => ";
                }
            }
            returnval ~= " ), ";
            
        }
        
        return returnval;
    }
    

}

class NumericSetSpace : discretefunctions.Set!(size_t) {


    public this(size_t start, size_t end) {
        Tuple!(size_t) [] tempArr;
        for (size_t i = start; i < end; i ++) {
            tempArr ~= tuple(i);
        } 

        super(tempArr);
    }

    public this (size_t count) {
        this(0, count);
    }
}

Set!(Tuple!(T)) pack_set(T...)(Set!(T) to_pack) {
    Tuple!(Tuple!(T)) [] tuple_arr = new Tuple!(Tuple!(T)) [to_pack.size()];

    size_t i = 0;
    foreach (a ; to_pack) {
        tuple_arr[i] = tuple(a);
        i ++;
    }
        
    return new Set!(Tuple!(T))(tuple_arr);

}

Function!(RETURN_TYPE, Tuple!(PARAM)) pack_function(RETURN_TYPE, PARAM...)(Function!(RETURN_TYPE, PARAM) to_pack) {

    auto space = pack_set(to_pack.param_set());

    RETURN_TYPE [Tuple!(Tuple!(PARAM))] arr;

    foreach(p, val ; to_pack.storage) {

        arr [ tuple(p) ] = val;
    }
    
    return new Function!(RETURN_TYPE, Tuple!(PARAM))(space, arr, to_pack.funct_default);
}


Distribution!(Tuple!(PARAM)) pack_distribution(PARAM...)(Distribution!(PARAM) to_pack) {

    auto f = pack_function(to_pack);

    auto returnval = new Distribution!(Tuple!(PARAM))(f);

    returnval.normalized = to_pack.normalized;
    returnval.normal = to_pack.normal;

    return returnval;
}


Set!(T) unpack_set(T...)(Set!(Tuple!(T)) to_unpack) {
    Tuple!(T) [] tuple_arr = new Tuple!(T) [to_unpack.size()];

    size_t i = 0;
    foreach (a ; to_unpack) {
        tuple_arr[i] = a[0];
        i ++;
    }
        
    return new Set!(T)(tuple_arr);

}

Function!(RETURN_TYPE, PARAM) unpack_function(RETURN_TYPE, PARAM...)(Function!(RETURN_TYPE, Tuple!(PARAM)) to_unpack) {

    auto space = unpack_set(to_unpack.param_set());

    RETURN_TYPE [Tuple!(PARAM)] arr;

    foreach(p, val ; to_unpack.storage) {

        arr [ p[0] ] = val;
    }    

    return new Function!(RETURN_TYPE, PARAM)(space, arr, to_unpack.funct_default);

}


Distribution!(PARAM) unpack_distribution(PARAM...)(Distribution!(Tuple!(PARAM)) to_unpack) {

    auto f = unpack_function(to_unpack);

    auto returnval = new Distribution!(PARAM)(f);

    returnval.normalized = to_unpack.normalized;
    returnval.normal = to_unpack.normal;

    return returnval;
}

// Specialized Distribution where each probability is proportional to e^param
class ExponentialDistribution(PARAMS...) {

    double [Tuple!PARAMS] params;
    double normalizer;
    
    public this(Distribution!(PARAMS) initial) {

        normalizer = 0.0;

        foreach(p ; initial.param_set()) {

            double val = log(initial[p[0]]);

            params[p] = val;
            normalizer += exp(val);
        }
        
    }

    double opIndex(Tuple!(PARAMS) i ) {
        double* p;
        p = (i in params);
        if (p !is null) {
            return exp(*p) / normalizer;
        }
/*        if ( mySet !is null && ! mySet.contains(i)) {
            throw new Exception("ERROR, key is not in the set this function is defined over.");
        }*/
        return 0.0;
    }

    void setParam(Tuple!(PARAMS) i, double newval) {
        double* p;
        p = (i in params);
        if (p !is null) {
            normalizer -= exp(*p);
            *p = newval;
        } else {
            params[i] = newval;
        }
        normalizer += exp(newval);
    }

    double getParam(Tuple!(PARAMS) i) {
        double* p;
        p = (i in params);
        if (p !is null) {
            return *p;
        }
        return 0.0;
    }

     public Tuple!(PARAMS) sample() {

        if (params.length == 0) {
            throw new Exception("Cannot sample from zero sized distribution.");
        }
          
        import std.random;

        auto rand = uniform(0.0, 1.0);

        auto mass = 0.0;
        foreach ( key, val; params) {
            mass += exp(val) / normalizer;

            if (mass >= rand)
                return key;
        }

/*        debug {
            import std.conv;
            throw new Exception("Didn't find a key to sample, ended at: " ~ to!string(mass) ~ " but wanted " ~ to!string(rand) ~ " " ~ to!string(params) ~ " " ~ to!string(normalizer));
        } else {*/
        Tuple!PARAMS returnval;
        foreach ( key, val; params) {
            returnval = key;
        }
     //   }
        return returnval;
    }

    override string toString() {

        string returnval = "";

        foreach (key, val ; params) {

            foreach (k ; key) {
                static if (isTuple!(typeof(k))) {
                    returnval ~= "(";
                    foreach(k2 ; k) {
                        returnval ~= to!string(k2) ~ ", ";
                    }
                    returnval ~= ") ";
                } else {
                    returnval ~= to!string(k) ~ ", ";
                }
            }
            
            returnval ~= " => " ~ to!string(exp(val) / normalizer) ~ ", ";
            
        }
        
        return returnval;
    }
            
}

Distribution!(T) set_to_uniform_probability(T...)(Set!T input_set, Set!T full_set) {

    Distribution!(T) returnval = new Distribution!T(full_set, 0.0);

    foreach (x; input_set) {

        returnval[x] = 1.0;
    }

    returnval.normalize();
    return returnval;
    
}

// a continuous distribution over discrete distributions, parameterized by alphas
class DirichletDistribution (PARAMS ...) {
    private double [Tuple!(PARAMS)] alphas;
    private double sumAlphas;
    private Set!PARAMS space;
    
    public this(double [Tuple!(PARAMS)] alphas, Set!PARAMS space, bool add_one_to_all_alphas = false) {
        this.space = space;
        setAlphas(alphas, add_one_to_all_alphas);
        
    }

    public double [Tuple!(PARAMS)] getAlphas() {
        return alphas.dup;
    }

    public Set!(PARAMS) param_set() {
        return space;
    }
    
    public void setAlphas(double [Tuple!(PARAMS)] alphas, bool add_one_to_all_alphas = false) {
        if (alphas.length != space.size()) {
            throw new Exception("Number of alpha parameters must match the number of variables in the set");
        }
        
        this.alphas = alphas.dup;
        if (add_one_to_all_alphas) {
            foreach (s ; space) {
                this.alphas[s] += 1;
            }
        }
        
        double sum = 0.0;
        foreach (a; this.alphas) {
            sum += a;
        }
        sumAlphas = sum;
    }

    public Distribution!(PARAMS) sample() {
        double [Tuple!(PARAMS)] arr;

        foreach (s; space) {

            arr[s] = gamma_sample(alphas[s], 1.0);
        }

        auto returnval = new Distribution!(PARAMS)(space, arr);
        returnval.normalize();

        return returnval;
    }

    public Distribution!(PARAMS) mean() {

        double [Tuple!(PARAMS)] arr;

        foreach (s; space) {

            arr[s] = alphas[s] / sumAlphas;
        }

        return new Distribution!(PARAMS)(space, arr);
        
    }

    public Distribution!(PARAMS) mode() {

        foreach (s ; space) {
            if (this.alphas[s] <= 1.0) {
                throw new Exception("Mode undefined for alphas <= 1.0: " ~ to!string(this.alphas));
            }
        }
            
        double [Tuple!(PARAMS)] arr;

        foreach (s; space) {

            arr[s] = (alphas[s] - 1) / (sumAlphas - space.size());
        }

        return new Distribution!(PARAMS)(space, arr);
        
    }

    private double normal_sample() {
        double total = 0;
        for (int i = 0; i < 12; i ++)  // irwin-hall approximation of the normal distribution 
           total += uniform01();
        return total - 6;
    }

    private double gamma_sample(double alpha, double beta) {

        if (alpha >= 1.0 ) {
            return beta * Marsaglia_gamma(alpha);
        } else {
            double gamma_a = Marsaglia_gamma(alpha + 1.0); 
            return beta * gamma_a * pow(uniform01(), 1.0 / alpha);
        }
    }

    private double Marsaglia_gamma(double alpha) {

        double d = alpha - (1.0 / 3);
        double c = 1.0 / sqrt(9 * d);

        do {
            double X = normal_sample();
            double v = pow((1 + c * X), 3);

            double u = uniform01();
            if (v > 0 && log(u) < ((X*X) / 2) + d - d*v + d * log (v))
                return d * v;
        } while(true);
        
    }

    double opIndex(Distribution!(PARAMS) i ) {
        import std.mathspecial;
        
        double numerator = 1.0;
        double denominator = 1.0;

        foreach (entry; space) {

            numerator *= pow(i[entry], alphas[entry] - 1);
            denominator *= gamma(alphas[entry]);            
        }

        denominator /= gamma(sumAlphas);
//import std.stdio;
//writeln("Dirichlet: " ~ to!string(alphas) ~ " " ~ to!string(sumAlphas) ~ " " ~ to!string(numerator) ~ " " ~ to!string(gamma(sumAlphas)));

        return numerator / denominator;
    }

    public void scale(double scale_val, double min = 1.0) {

        double newSumAlphas = 0.0;
        
        foreach (entry; space) {

//            alphas[entry] = fmax(min, (scale_val / mag_alphas) * alphas[entry]);
            alphas[entry] = fmax(min, (scale_val / sumAlphas) * alphas[entry]);
            newSumAlphas += alphas[entry]; 
        }

        sumAlphas = newSumAlphas;
    }

    public void addAlphas(double [Tuple!(PARAMS)] newVals) {
        double [Tuple!(PARAMS)] tempVals;

        foreach (entry; space) {

            double a = 0.0;
            double b = 0.0;
            double * p;

            p = entry in alphas;
            if (p !is null)
                a = *p;
            p = entry in newVals;
            if (p !is null)
                b = *p;

            tempVals[entry] = a + b;
        }

        setAlphas(tempVals);
        
    }

    public void interpolateAlphas(double [Tuple!(PARAMS)] newVals, double otherPercent) {
        double [Tuple!(PARAMS)] tempVals;

        foreach (entry; space) {

            double a = 0.0;
            double b = 0.0;
            double * p;

            p = entry in alphas;
            if (p !is null)
                a = *p;
            p = entry in newVals;
            if (p !is null)
                b = *p;

            tempVals[entry] = ((1.0 - otherPercent) * a) + (otherPercent * b);
        }

        setAlphas(tempVals);
        
    }

    public double alpha_sum() {
        return sumAlphas;
    }

    override string toString() {
        string returnval = "";
        foreach (k, v; alphas) {
            returnval ~= to!string(k[0]) ~ " => " ~ to!string(v) ~ ", ";
        }
        return returnval;
    }
}

class DirichletProcess (T) {

    private Distribution!(T) host_distribution;

    private long [Tuple!T] sample_count;
    private long n;
    private double alpha;
    

    
    public this (Distribution!(T) host_distribution, double alpha) {

        if (alpha <= 0)
            throw new Exception("Alpha must be a positive number");
            
        this.host_distribution = host_distribution;
        n = 0;
        this.alpha = alpha;

        
    }

    public Tuple!T peek() {
        import std.random;


        auto pick = uniform01();
        double denom = alpha + n;
        double mass = alpha / denom;

        if (n < 1 || mass >= pick) {

            return host_distribution.sample();
        }

        Tuple!T returnval;
        foreach ( key, val; sample_count) {
            mass += val / denom;

            if (mass >= pick)
                return key;
            returnval = key;
        }

        return returnval;        
    }

    public void increment(Tuple!T observation) {

        long * p;
        p = (observation in sample_count);
        if (p !is null) {
            *p += 1;
        } else {
            sample_count[observation] = 1;
        }

        n++;
    }


    public Tuple!T sample () {

        auto returnval = peek();

        increment(returnval);

        return returnval;

    }

    public double probabilityOf(Tuple!T observation) {

        double denom = alpha + n;
        
        return host_distribution[observation] * (alpha / denom) + 
                sample_count.get(observation, 0) / denom;
    }

}

