import {hash, compare} from "bcryptjs";
import { auth } from "~/server/auth";


export async function hashPassword(password: string) {

    return hash(password, 12);

}

// export async function getCurrentUser() {
//     const serverSession = await auth();
//     const user = serverSession?.user;
//     if (!user) {
//         throw new Error("User not authenticated");
//     }
//     return user;
// }


export async function verifyPassword(password: string, hashedPassword: string) {

    return compare(password, hashedPassword);

}