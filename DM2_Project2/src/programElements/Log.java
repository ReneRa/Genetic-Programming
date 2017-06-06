package programElements;

public class Log extends Operator{

	private static final long serialVersionUID = 7L;
	
	public Log(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
		return Math.round(Math.log(arguments[0]));
	}
	
	public String toString() {
		return "log";
	}
	
	
}
