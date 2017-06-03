package programElements;

public class randDiv extends Operator{
	private static final long serialVersionUID = 7L;
	
	public randDiv(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
				return (arguments[0] / Math.random());
	}
	
	public String toString() {
		return "randDiv";
	}
}



