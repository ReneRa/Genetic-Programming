package programElements;

//import utils.Utils;
import java.lang.Math;

public class aSquare extends Operator {

	private static final long serialVersionUID = 7L;
	
	public aSquare(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
				return Math.pow(arguments[0], 2);
		//		return Utils.aPowerB(arguments[0], arguments[1]);
	}
	
	public String toString() {
		return "a^2";
	}
}