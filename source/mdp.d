module mdp;


class Space {

}


// maps from a space to a real number
class Function (RETURN_TYPE, PARAM ...) {


    abstract Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1]) max();

    abstract PARAM argmax();

    abstract Function!(RETURN_TYPE, PARAM[0 .. PARAM.length - 1]) sumout();

    // This = (S, A) => R , funct = S => A, returns S => R
    abstract Function!(RETURNTYPE, B) apply(A : PARAM[PARAM.length - 1], B : PARAM[0 .. PARAM.length - 1] )(Function!(A, B) funct);
    
}

class Model {

}


enum DistInitType {None, Uniform, Exponential, RandomFromGaussian};
