module partialobservation;

import maxentIRL;
import discretemdp;
import discretefunctions;
import std.typecons;

class Observation {

}


class PartialObservabilityIRL_Shervin  : Sequence_MaxEnt_Problem!(State, Action, Observation) {

    protected Sequence_MaxEnt_Problem!(State, Action) underlying_solver;

    public this(Sequence_MaxEnt_Problem!(State, Action) solver) {
        this.underlying_solver = solver;
    }

    public double [] solve (Sequence!(Distribution!(State, Action, Observation)) [] trajectories, bool stochasticGradientDescent = true) {

        // convert the trajectories from Pr(S, A, o) => Pr(S, A)

        Sequence!(Distribution!(State, Action)) [] reduced_trajectories = new Sequence!(Distribution!(State, Action)) [trajectories.length]; 


        foreach (i, traj ; trajectories) {

            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length);

            foreach(t, timestep; traj) {

                seq[t] = tuple(new Distribution!(State, Action)( sumout(timestep[0]) ) );
            }

            reduced_trajectories[i] = seq;
            
        }
        
        return underlying_solver.solve(reduced_trajectories, stochasticGradientDescent);
    }
    
}

