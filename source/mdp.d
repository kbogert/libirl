module mdp;

import std.meta;


class Space {

}


// maps from a space to a real number
class Function (RETURN_TYPE, PARAM ...) {


    abstract Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 2] ) max();

    abstract Function!(RETURN_TYPE, removeLast(ToRemove) ) max(ToRemove ...)();

    // this should return a function that maps some of params to others, so argmax(Action) on Q(S,A) => R returns a policy S => A
    abstract tuple!PARAM argmax();

    abstract Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 2] ) sumout();
    
    abstract Function!(RETURN_TYPE, removeLast(ToRemove) ) sumout(ToRemove ...)();

    // This = (S, A) => R , funct = S => A, returns S => R
    abstract Function!(RETURN_TYPE, B) apply(A : PARAM[PARAM.length - 1], B : PARAM[0 .. PARAM.length - 2] )(Function!(A, B) funct);


    protected template removeLast(FIRST, T ...) {
        static if (T.length > 0) {
            auto removeLast = Reverse!(Erase!(FIRST, Reverse!(  removeLast(T) )));
        } else {
            auto removeLast = Reverse!(Erase!(FIRST, Reverse!(PARAM)));
        }
    }
}

class Model {

}


enum DistInitType {None, Uniform, Exponential, RandomFromGaussian};
