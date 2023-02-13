-- l cardinal = f in (a in (b in f b a));
cardinal f a b = f b a;

-- l kestrel = a in (b in a);
kestrel a b = a

main :: IO ()
--main = putStrLn (show (cardinal id 1 2)) -- can't apply `2` to `1` because:
-- cardinal id 1 2 = id 2 1 = 2 1
-- 2 1 isnt a valid function call in haskell

main = putStrLn (show (cardinal kestrel 1 id 2))
