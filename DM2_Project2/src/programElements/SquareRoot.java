package programElements;

//import utils.Utils;
import java.lang.Math;

public class SquareRoot extends Operator{

	private static final long serialVersionUID = 7L;
	
	public SquareRoot(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
		return Math.round(Math.sqrt(arguments[0]));
	}	
	
	public String toString() {
		return "sqrt";
	}

}


	
