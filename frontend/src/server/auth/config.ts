import { PrismaAdapter } from "@auth/prisma-adapter";
import { type DefaultSession, type NextAuthConfig } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { verifyPassword } from "~/lib/auth";
import { db } from "~/server/db";
import Google from "next-auth/providers/google";
import { env } from "~/env";
// import Stripe from "stripe";
import axios from "axios";

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

}


const CustomPrismaAdapter = PrismaAdapter(db);
CustomPrismaAdapter.createUser = async (data) => {
  console.log("inside custom adapter")
  const { password, ...userData } = data  ; // Exclude password if not provided

    // const stripe = new Stripe(env.STRIPE_SECRET_KEY);
  
    // const stripeCustomer = await stripe.customers.create({
    //   email: data.email.toLowerCase(),
    // });

    const razorpayResponse = await axios.post("https://api.razorpay.com/v1/customers", {
      email: userData.email.toLowerCase(),
      name: userData.name || userData.email.split("@")[0],
      type: "customer",
      
    }, {
      auth: {
        username: env.RAZORPAY_KEY_ID,
        password: env.RAZORPAY_KEY_SECRET,
      },
    });

  // return db.user.create({
  //   data: {
  //     ...userData,
  //     stripeCustomerId: stripeCustomer.id,
  //   },
  // });



    return db.user.create({
        data: {
          ...userData,
          razorpayContactId: razorpayResponse.data.id, // Store Razorpay contact ID
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
