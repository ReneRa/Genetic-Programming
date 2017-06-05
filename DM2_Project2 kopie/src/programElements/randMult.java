package programElements;

public class randMult extends Operator{
	private static final long serialVersionUID = 7L;
	
	public randMult(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
				return (arguments[0] * Math.random());
	}
	
	public String toString() {
		return "randMult";
	}
}
