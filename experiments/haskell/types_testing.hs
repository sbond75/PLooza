id' x = x

main :: IO ()
main = putStrLn ((show (id' 1)) ++ (show (id' "1"))) -- prints: `1"1"`
