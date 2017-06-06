package programElements;

public class Log10 extends Operator{

	private static final long serialVersionUID = 7L;
	
	public Log10(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
		return Math.round(Math.log10(arguments[0]));
	}
	
	public String toString() {
		return "log10";
	}
	
}
