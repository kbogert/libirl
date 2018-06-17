module utility;

import std.algorithm;
import std.math;
        
double l1norm(double [] arr) {

    return reduce!((a,b) => a + abs(b))(0.0, arr);
}

double l2norm(double [] arr) {

    return sqrt(sumSquares(arr));
}

double sumSquares(double [] arr) {
    return reduce!((a,b) => a + b * b)(0.0, arr);
}
