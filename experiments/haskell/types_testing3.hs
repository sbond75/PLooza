-- l mockingbird = f in (f f);
mockingbird f = f f;

main :: IO ()
main = putStrLn (show mockingbird)
