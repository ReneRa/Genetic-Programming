package programElements;

public class Power extends Operator {

	private static final long serialVersionUID = 7L;
	
	public Power(){
		super (2);
	}
	
	public double performOperation(double... arguments) {
		return Math.round(Math.pow(arguments[0], arguments[1]));
	}
	
	public String toString() {
		return "^";
	}
}
	
