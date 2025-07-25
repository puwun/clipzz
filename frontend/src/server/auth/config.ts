import { PrismaAdapter } from "@auth/prisma-adapter";
import { type DefaultSession, type NextAuthConfig } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { verifyPassword } from "~/lib/auth";
import { db } from "~/server/db";
import Google from "next-auth/providers/google";
import { env } from "~/env";
import Stripe from "stripe";

/**
 * Module augmentation for `next-auth` types. Allows us to add custom properties to the `session`
 * object and keep type safety.
 *
 * @see https://next-auth.js.org/getting-started/typescript#module-augmentation
 */
declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      // ...other properties
      // role: UserRole;
    } & DefaultSession["user"];
  }

  // interface User {
  //   // ...other properties
  //   // role: UserRole;
  // }
}


const CustomPrismaAdapter = PrismaAdapter(db);
CustomPrismaAdapter.createUser = async (data) => {
  console.log("inside custom adapter")
  const { password, ...userData } = data  ; // Exclude password if not provided

    const stripe = new Stripe(env.STRIPE_SECRET_KEY);
  
    const stripeCustomer = await stripe.customers.create({
      email: data.email.toLowerCase(),
    });
  

    console.log("---------------------------before creating google user ", password, stripeCustomer.id)



  return db.user.create({
    data: {
      ...userData,
      stripeCustomerId: stripeCustomer.id,
    },
  });
};


/**
 * Options for NextAuth.js used to configure adapters, providers, callbacks, etc.
 *
 * @see https://next-auth.js.org/configuration/options
 */
export const authConfig = {
  providers: [
    CredentialsProvider({
      
      name: "credentials",
      
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },

      async authorize(credentials) {
        
        if (!credentials?.email || !credentials?.password) {
          return null;
        }
        
        const email = credentials.email as string;
        const password = credentials.password as string;

        const user = await db.user.findUnique({
          where: { email },
        });


        // console.log("User found:", user);
        // console.log("Email:", email);
        // console.log("Password:", password);

        if(!user){
          return null;
        }

        const isValidPassword = await verifyPassword(password, user.password);

        if (!isValidPassword) {
          return null;
        }

        return user;
      }
    }),
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    })
  ],
  session : { strategy: "jwt" }, 
  adapter: CustomPrismaAdapter,
  callbacks: {
    session: ({ session, token }) => ({
      ...session,
      user: {
        ...session.user,
        id: token.sub,
      },
    }),
    jwt: ({ token, user }) => {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
  },
} satisfies NextAuthConfig;
