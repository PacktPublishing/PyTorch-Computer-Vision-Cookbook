## List of Errors/Typos

These errors and typos were found and reported by the readers. Thank you all for reproing them.



- Page 130: in the bottom, the line h, w = img.size, it should be w, h = img.size 

- Page 135: in the bottom, the line label = x, w-y, it should be label = x, h-y 

- On Page 231, the type of the img and mask are torch.FloatTensor which means the dtype should be torch.float32, while on Page 234, 
the dtype of the img and mask are torch.uint8, they should both be torch.float32

- On Page 234, the shape of img and mask are [16, 1, 128, 192] which is a 4-D tensor, 
while on Page 236, it states that the shape are [batch_size, 128, 192] which is a 3-D tensor, 
on Page 245, img and mask are treat as a 3-D tensor because the line xb=xb.unsqueeze(1).type(torch.float32).to(device).
They should both be 4-D tensors since they are from dataloader output. The errors are in the code part of the book. But the repo scripts are correct.
Please refer to the repo scripts for correct implementation.

- Page 307, on the top, fixed_noise=torch.randn(16, nz, 1, 1, device=device), img_fake=model_gen(noise).detach().cpu(). 
They should be either fixed_noise or noise be.

- Page 330, on the top, when model_type='3dcnn', the collate_fn was called collate_fn_r3d_18, but the definition was collate_fn_3dcnn. There is a typo here.
They should be the same. 

- Page 331, on the top, the shape of the input of the model when model_type='rnn' is [16, 3, 16, 112, 112], there is typo here.
it should be [16, 16, 3, 112, 112].



* Page 187 (Bullet point 1, second code block): 

**# Remove empty boxes**

**targets = [boxes for boxes in targets if boxes is not None]**  is logically incorrect but it is valid for this example.

**Explaination**: If there is any target that is None, the function will remove it first, resulting in a list with fewer than 8 elements. Then when we assign the index to every element in the list, the index will not reflect the order of the images, because we have already removed some target. For example:

`targets = [bbox0, bbox1, bbox2, None, bbox4, bbox5, bbox6, bbox7]`

after removing the None element:

`targets = [bbox0, bbox1, bbox2, bbox4, bbox5, bbox6, bbox7]`

then we assign them the index which should be (indexes are in the parentheses):

`targets = [bbox0 (0), bbox1 (1), bbox2 (2), bbox4 (4), bbox5 (5), bbox6 (6), bbox7 (7)]`

but the program now will assign (index are in the parentheses):

`targets = [bbox0 (0), bbox1 (1), bbox2 (2), bbox4 (3), bbox5 (4), bbox6 (5), bbox7 (6)]`

The reason it is valid here, is, for the dataset used in the chapter, there is no target that is `None`, so the function will not remove any element from the list. 
