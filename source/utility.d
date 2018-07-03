module utility;

import std.algorithm;
import std.math;
import std.array;
        
double l1norm(double [] arr) {

    return reduce!((a,b) => a + abs(b))(0.0, arr);
}

double l2norm(double [] arr) {

    return sqrt(sumSquares(arr));
}

double sumSquares(double [] arr) {
    return reduce!((a,b) => a + b * b)(0.0, arr);
}


double [] clamp(double [] arr, double min_d, double max_d) {

    double [] returnval = arr.dup;
    
    foreach (ref a; returnval) {
        a = std.algorithm.clamp(a, min_d, max_d);
    }

    return returnval;
}

double [] array_abs(double [] data) {

    double [] returnval = data.dup;

    foreach (ref a; returnval) {
        a = abs(a);
    }
        
    return returnval;
}

double abs_average(double [][] data) {

    if (data.length == 0)
        return double.infinity;
        
    double [] sum = minimallyInitializedArray!(double[])(data[0].length);
    foreach (entry; data) {
        sum [] += array_abs(entry)[];
    }
    sum [] /= data.length;
    
    double returnval = 0;
    foreach (s ; sum) {
        returnval += s;
    }
    return returnval / sum.length;

}

double abs_diff_average(double [] data) {

    if (data.length == 0)
        return double.infinity;
    
    double returnval = 0;

    foreach (i; 0 .. data.length) {
        returnval += abs( data[i] - data[ (i + 1) % data.length ] );
    }
    returnval /= data.length;

    return returnval;
    
}

// from: https://forum.dlang.org/post/kfg93v$2lju$1@digitalmars.com
double stddev(double [] data) {
    auto n = data.length;
    auto avg = reduce!((a, b) => a + b / n)(0.0, data);
    auto var = reduce!((a, b) => a + pow(b - avg, 2) / n)(0.0, data);

    return sqrt(var);         
}
