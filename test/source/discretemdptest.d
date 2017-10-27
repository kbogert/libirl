import discretemdp;
import tested;
import std.conv;
import std.math;

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

     public size_t size() {
         return arr.length;
     }

     public bool contains(testObj i) {
         foreach( a ; arr) 
             if (a == i)
                 return true;
         return false;
     }

     int opApply(int delegate(ref testObj) dg) {
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

   dist = new Distribution!(testObj)(tos, mdp.DistInitType.Uniform);

   assert(dist.size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.size()) ~ " should be: " ~ to!string(spaceSize));

   foreach(key, val ; dist) {
       assert(val == 1.0 / cast(double)spaceSize, "Probability incorrect in uniform distribution");
   }

   debug {
       import std.stdio;
       writeln(dist);
   }

   dist = new Distribution!(testObj)(tos, mdp.DistInitType.RandomFromUniform);

   assert(dist.size() == spaceSize, "Distribution size is incorrect: " ~ to!string(dist.size()) ~ " should be: " ~ to!string(spaceSize));
   double total = 0;
   foreach(key, val ; dist) {
       total += val;
   }

   debug {
       import std.stdio;
       writeln(dist);
   }
   assert(abs(1.0 - total) < 0.000000001, "Probability distribution not normalized: " ~ to!string(total) ~ " should be 1.0");

   dist = new Distribution!(testObj)(tos, mdp.DistInitType.RandomFromGaussian);

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

   assert(abs(1.0 - total) < 0.000000001, "Probability distribution not normalized: " ~ to!string(total) ~ " should be 1.0");

   
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

   debug {
       import std.stdio;
       writeln();
       writeln(dist);
   }

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

   debug {
       import std.stdio;
       writeln();
       writeln(dist);
   }


   dist.normalize();

   assert (dist.argmax() == new testObj(0), "Argmax didn't work!");



}


@name("Distribution iterations")
unittest {

    assert(true);
}
