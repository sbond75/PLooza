-- l kestrel = a in (b in a);
kestrel a b = a

-- l true_ = kestrel; // selects the first expression given
true_ = kestrel

-- l false_ = id; // selects the second expression given
false_ = kestrel id

-- l not = p in ((p false_) true_);
not p = p false_ true_;

main :: IO ()
main = putStrLn (show ((Main.not true_) True False)) -- prints False
