import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Random;

public class AsciiPrime {

    private char[] number;
    private final int PRIME_CERTAINTY = 100;
    private final char FILL_DIGIT = 9;
    private final char[] POSSIBLE_DIGITS = {0, 3, 5, 6, 8, 8, 8};
    private ArrayList<Integer> fill_positions;
    private int width, height;
    private Random rnd = new Random();


    public AsciiPrime(String prime) {
        fill_positions = new ArrayList<>();
        String[] rows = prime.split("\n");
        height = rows.length;
        width = rows[0].length();
        number = new char[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                assert rows[y].length() == width : String.format("row %d length was %d, but should be %d", y, rows[y].length(), width);
                char cur_number = rows[y].charAt(x);
                number[y * width + x] = cur_number;
                if (cur_number - '0' == FILL_DIGIT)
                    fill_positions.add(y * width + x);
            }
    }

    boolean isPrime() {
        BigInteger intNumber = new BigInteger(new String(number));
        return intNumber.isProbablePrime(PRIME_CERTAINTY);
    }

    void changeRandomPos() {
        int pos = fill_positions.get(rnd.nextInt(fill_positions.size()));
        char fill_digit;
        while (true) {
            fill_digit = (char) (POSSIBLE_DIGITS[rnd.nextInt(POSSIBLE_DIGITS.length)] + '0');
            if (number[pos] != fill_digit)
                break;
        }
        number[pos] = fill_digit;
    }

    public String toString() {
        StringBuilder builder = new StringBuilder(width * height + height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                builder.append(number[y * width + x] - '0');
            }
            builder.append("\n");
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        String input = "20181111111111111111111111111111111111\n" +
                "11111111111111111199111111111111111111\n" +
                "11111111111111111999911111111111111111\n" +
                "11111111111111119999991111111111111111\n" +
                "11111111111111199999999111111111111111\n" +
                "11111111111111999999999991111111111111\n" +
                "11111111111199999999999999111111111111\n" +
                "11111111111111199999999111111111111111\n" +
                "11111111111111999999999911111111111111\n" +
                "11111111111119999999999991111111111111\n" +
                "11111111111999999999999999911111111111\n" +
                "11111111119999999999999999991111111111\n" +
                "11111111999999999999999999999911111111\n" +
                "11111199999999999999999999999999111111\n" +
                "11111111111119999999999999111111111111\n" +
                "11111111111999999999999999911111111111\n" +
                "11111111119999999999999999991111111111\n" +
                "11111111999999999999999999999911111111\n" +
                "11111199999999999999999999999999111111\n" +
                "11119999999999999999999999999999991111\n" +
                "11999999999999999999999999999999999911\n" +
                "11111111111111111999911111111111111111\n" +
                "11111111111111111999911111111111111111\n" +
                "11111111111111111999911111111111111111";

        AsciiPrime prime = new AsciiPrime(input);
        int iterations = 0;
        while (!prime.isPrime()) {
            prime.changeRandomPos();
            iterations++;
            if ((iterations % 50) == 0)
                System.out.println("Iteration: " + iterations);
        }
        System.out.println("Iteration: " + iterations);
        System.out.println(prime);
    }
}
