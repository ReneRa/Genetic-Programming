package programElements;

public class Min extends Operator{
	private static final long serialVersionUID = 7L;
	
	public Min(){
		super (2);
	}
	
	public double performOperation(double... arguments) {
				return Math.min(arguments[0], arguments[1]);
	}
	
	public String toString() {
		return "min";
	}
}
