module partialobservation;

import maxentIRL;
import discretemdp;
import discretefunctions;
import std.typecons;
import std.conv;

class Observation {

}

class GenericObservation : Observation {

    private int id;

    public this(int i) {
        id = i;
    }

    public int getId() {
        return id;
    }

    public override string toString() {
        return "Observation: " ~ to!string(id);
    }

    override bool opEquals(Object o) {
        auto rhs = cast(GenericObservation)o;
        if (!rhs) return false;

        return id == rhs.id;
    }

    override size_t toHash() @trusted nothrow {
        return id;
    }    
}


class PartialObservabilityIRL_Shervin  : Sequence_MaxEnt_Problem!(State, Action, Observation) {

    protected Sequence_MaxEnt_Problem!(State, Action) underlying_solver;

    public this(Sequence_MaxEnt_Problem!(State, Action) solver) {
        this.underlying_solver = solver;
    }

    public double [] solve (Sequence!(Distribution!(State, Action, Observation)) [] trajectories) {

        // convert the trajectories from Pr(S, A, o) => Pr(S, A)

        Sequence!(Distribution!(State, Action)) [] reduced_trajectories = new Sequence!(Distribution!(State, Action)) [trajectories.length]; 


        foreach (i, traj ; trajectories) {

            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length);

            foreach(t, timestep; traj) {

                seq[t] = tuple(new Distribution!(State, Action)( sumout(timestep[0]) ) );
            }

            reduced_trajectories[i] = seq;
            
        }
        
        return underlying_solver.solve(reduced_trajectories);
    }
    
}

