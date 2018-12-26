package util;

import java.util.Random;

public class CommonUtils {
    public static int randomInt(int leftLimit, int rightLimit){
        return leftLimit + (int) (new Random().nextFloat() * (rightLimit - leftLimit));
    }
}
