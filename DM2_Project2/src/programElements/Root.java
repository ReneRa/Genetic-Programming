package programElements;

//import utils.Utils;
import java.lang.Math;

public class Root extends Operator {

	private static final long serialVersionUID = 7L;
	
	public Root(){
		super (2);
	}
	
	public double performOperation(double... arguments) {
				return Math.round(Math.pow(arguments[0], (1/arguments[1])));
	}
	
	public String toString() {
		return "root";
	}
}